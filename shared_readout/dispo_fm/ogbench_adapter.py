"""Offline DiSPO training and goal evaluation for OGBench AntMaze."""

from pathlib import Path

import numpy as np
import torch

from .core import Replay
from .features import fourier_features


DATASETS = tuple(f'antmaze-medium-{kind}-v0' for kind in ('navigate', 'stitch', 'explore'))
TRAIN_UPDATES = 100_000
EVAL_UPDATES = 5_000
EVAL_EPISODES = 10


def open_env(dataset):
    import ogbench

    return ogbench.make_env_and_datasets(dataset, env_only=True, terminate_at_goal=False)


def load_replay(dataset, device):
    from ogbench.utils import DEFAULT_DATASET_DIR, download_datasets, load_dataset

    download_datasets([dataset])
    data = load_dataset(Path(DEFAULT_DATASET_DIR).expanduser() / f'{dataset}.npz')
    observations = data['observations']
    replay = Replay()
    replay.size = replay.capacity = len(observations)
    replay.arrays = dict(
        observations=observations, actions=data['actions'],
        features=fourier_features(observations, device),
        next_observations=data['next_observations'],
        terminated=np.zeros(replay.size, dtype=bool), truncated=data['terminals'].astype(bool),
    )
    return replay


def epoch_batches(size, batch_size, seed, start=0):
    """One seeded shuffle per epoch; skip the tail and restore the batch cursor."""
    steps = size // batch_size
    if steps == 0:
        raise ValueError('Batch size exceeds the training dataset')
    epoch, offset = divmod(start, steps)
    while True:
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        permutation = rng.permutation(size)
        for batch in range(offset, steps):
            yield permutation[batch * batch_size:(batch + 1) * batch_size]
        epoch += 1
        offset = 0


@torch.no_grad()
def evaluate_antmaze(learner, spec, episodes, seed):
    env = open_env(spec['ogbench_dataset'])
    rng = torch.Generator(device=learner.device).manual_seed(seed)
    returns = np.zeros(episodes)
    successes = np.zeros(episodes, dtype=bool)
    goal_ids = 1 + np.arange(episodes) % env.unwrapped.num_tasks
    horizon = spec['episode_steps']
    try:
        indices = np.random.default_rng(seed).choice(
            learner.replay.size, min(10_000, learner.replay.size), replace=False)
        next_obs = learner.replay.arrays['next_observations'][indices]
        design = fourier_features(next_obs, learner.device).astype(np.float64)
        for i, task_id in enumerate(goal_ids):
            # The official reset also uses NumPy's legacy global RNG for XY jitter.
            state = np.random.get_state()
            try:
                np.random.seed((seed + i) % (2**32))
                env.action_space.seed(seed + i)
                obs, _ = env.reset(seed=seed + i, options={'task_id': int(task_id)})
            finally:
                np.random.set_state(state)
            goal = env.unwrapped.cur_goal_xy
            reward_targets = np.exp(-np.linalg.norm(next_obs[:, :2] - goal, axis=1) / 20)
            weights = learner.tensor(np.linalg.lstsq(design, reward_targets, rcond=None)[0])
            guidance = .5 * weights * learner.flow.span * .5
            for _ in range(horizon):
                observation = learner.tensor(obs)[None]
                outcome = learner.flow.sample(observation, rng, guidance=guidance)[0]
                action = learner._action(observation, outcome, rng)
                obs, _, terminated, truncated, info = env.step(action)
                returns[i] += np.exp(-np.linalg.norm(obs[:2] - goal) / 20)
                successes[i] |= bool(info['success'])
                if terminated or truncated:
                    break
    finally:
        env.close()
    row = dict(round=learner.round, updates=learner.flow.steps, mean_return=float(returns.mean()),
               success_rate=float(successes.mean()), horizon=horizon, candidates=1,
               protocol='DiSPO-style guided FM rollouts with dense AntMaze reward')
    return row, dict(returns=returns, success=successes, goal_ids=goal_ids)
