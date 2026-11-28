"""Offline DiSPO training and goal evaluation for OGBench AntMaze."""

from pathlib import Path

import numpy as np
import torch

from .core import Replay
from .features import fourier_features


DATASETS = tuple(f'antmaze-medium-{kind}-v0' for kind in ('navigate', 'stitch', 'explore'))
TRAIN_UPDATES = 100_000
EVAL_UPDATES = 5_000
EVAL_EPISODES = 250


def open_env(dataset, terminate_at_goal=False):
    import ogbench

    return ogbench.make_env_and_datasets(dataset, env_only=True, terminate_at_goal=terminate_at_goal)


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
    """Official task resets and success termination, with batched FM inference."""
    from tqdm import tqdm

    rng = torch.Generator(device=learner.device).manual_seed(seed)
    returns = np.zeros(episodes)
    dense_returns = np.zeros(episodes)
    successes = np.zeros(episodes, dtype=bool)
    lengths = np.zeros(episodes, dtype=np.int32)
    goal_ids = 1 + np.arange(episodes) % 5
    horizon = spec['episode_steps']
    indices = np.random.default_rng(seed).choice(
        learner.replay.size, min(10_000, learner.replay.size), replace=False)
    next_obs = learner.replay.arrays['next_observations'][indices]
    design = fourier_features(next_obs, learner.device).astype(np.float64)
    # The regression design is shared across all episode goals.
    inverse_design = np.linalg.pinv(design)
    with tqdm(total=episodes, desc='OGBench evaluation', unit='episode') as progress:
        for start in range(0, episodes, 50):
            stop = min(start + 50, episodes)
            environments, observations, goals = [], [], []
            try:
                for i in range(start, stop):
                    env = open_env(spec['ogbench_dataset'], terminate_at_goal=True)
                    environments.append(env)
                    # OGBench also uses NumPy's global RNG for reset XY jitter.
                    state = np.random.get_state()
                    try:
                        np.random.seed((seed + i) % (2**32))
                        env.action_space.seed(seed + i)
                        obs, _ = env.reset(seed=seed + i, options={'task_id': int(goal_ids[i])})
                    finally:
                        np.random.set_state(state)
                    observations.append(obs)
                    goals.append(env.unwrapped.cur_goal_xy)
                observations = np.asarray(observations, dtype=np.float32)
                goals = np.asarray(goals)
                reward_targets = np.exp(
                    -np.linalg.norm(next_obs[:, None, :2] - goals[None], axis=2) / 20)
                weights = learner.tensor((inverse_design @ reward_targets).T)
                guidance = .5 * weights * learner.flow.span * .5
                active = np.ones(stop - start, dtype=bool)
                for _ in range(horizon):
                    observation = learner.tensor(observations)
                    outcomes = learner.flow.sample(observation, rng, guidance=guidance)
                    context = torch.cat((observation, outcomes), dim=1)
                    normalized = learner.readout.sample(context, rng)
                    actions = (learner.action_low + (normalized + 1) * .5 * learner.action_span).cpu().numpy()
                    actions = np.clip(actions, -1, 1)
                    for j in np.flatnonzero(active):
                        obs, reward, terminated, truncated, info = environments[j].step(actions[j])
                        observations[j] = obs
                        i = start + j
                        returns[i] += reward
                        dense_returns[i] += np.exp(-np.linalg.norm(obs[:2] - goals[j]) / 20)
                        lengths[i] += 1
                        if terminated or truncated:
                            successes[i] = bool(info['success'])
                            active[j] = False
                            progress.update()
                    if not active.any():
                        break
                # For an explicitly shortened diagnostic horizon, retain the
                # final step's success value as the official evaluator does.
                for j in np.flatnonzero(active):
                    successes[start + j] = bool(environments[j].unwrapped.compute_success())
                    progress.update()
            finally:
                for env in environments:
                    env.close()
    task_success = {str(task): float(successes[goal_ids == task].mean())
                    for task in np.unique(goal_ids)}
    row = dict(round=learner.round, updates=learner.flow.steps, mean_return=float(returns.mean()),
               success_rate=float(np.mean(list(task_success.values()))),
               task_success_rates=task_success, horizon=horizon, candidates=1,
               protocol='OGBench official goal resets and success termination; mean success over tasks')
    return row, dict(returns=returns, dense_returns=dense_returns, success=successes,
                     goal_ids=goal_ids, lengths=lengths)
