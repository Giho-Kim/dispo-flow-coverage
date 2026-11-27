"""Offline DiSPO training and rollout evaluation for OCBench state observations."""

import numpy as np
import torch

from .core import Replay, elite_mean


DEFAULT_TASK = 'block-cpu-lite-double-task2-v0'


def open_env(task):
    if task.startswith('visual-'):
        raise ValueError('Use an OCBench task with state observations')
    import ocbench

    family, backend, _, _ = ocbench.parse_env_spec(task)
    # The CPU counterpart exposes the same per-world observation and action spaces.
    env = ocbench.make(task if backend == 'cpu' else task.replace(f'{family}-', f'{family}-cpu-', 1))
    if len(env.observation_space.shape) != 1 or len(env.action_space.shape) != 1:
        env.close()
        raise ValueError('OCBench requires flat observations and continuous actions')
    return env


def load_ocbench_replay(path, task, shards):
    from ocbench.dataset_utils import download_datasets, get_episode_offsets, load_dataset

    paths = [path] if path is not None else download_datasets(task, num_shards=shards)[0]
    data = load_dataset(paths)
    if int(data['observation_interval']) != 1:
        raise ValueError('DiSPO needs OCBench observations at every action step')
    action_offsets, obs_offsets = get_episode_offsets(data)
    lengths = np.diff(action_offsets)
    size = int(action_offsets[-1])
    if size == 0:
        raise ValueError('OCBench training dataset is empty')
    # Each episode stores one more observation than action.
    source = np.arange(size) + np.repeat(obs_offsets[:-1] - action_offsets[:-1], lengths)
    obs = np.asarray(data['observations'][source], dtype=np.float32)
    next_obs = np.asarray(data['observations'][source + 1], dtype=np.float32)
    rewards = np.asarray(data['rewards'], dtype=np.float32).reshape(size, 1)
    if not np.all((rewards >= 0) & (rewards <= 1)):
        raise ValueError('Expected binary or bounded OCBench task rewards')
    terminated = np.asarray(data['masks']).reshape(size) == 0
    truncated = np.asarray(data['terminals'], dtype=bool).reshape(size) & ~terminated
    replay = Replay()
    replay.arrays = dict(
        observations=obs, actions=np.asarray(data['actions'], dtype=np.float32),
        features=np.concatenate((np.tanh(obs), rewards), axis=1),
        next_observations=next_obs, terminated=terminated, truncated=truncated,
    )
    if not all(np.isfinite(array).all() for array in replay.arrays.values()):
        raise ValueError('OCBench dataset contains nonfinite values')
    replay.size = replay.capacity = size
    return replay


@torch.no_grad()
def evaluate_ocbench(learner, spec, candidates, episodes, seed):
    task = spec['ocbench_task']
    rng = torch.Generator(device=learner.device).manual_seed(seed)
    returns = np.zeros(episodes)
    success = np.zeros(episodes, dtype=bool)
    horizon = spec['episode_steps']
    if '-cpu-' not in task:
        evaluate_mjwarp(learner, task, candidates, horizon, seed, rng, returns, success)
        simulator = 'mjwarp-cuda'
    else:
        evaluate_cpu(learner, task, candidates, horizon, seed, rng, returns, success)
        simulator = 'mujoco-cpu'
    row = dict(round=learner.round, mean_return=float(returns.mean()),
               success_rate=float(success.mean()), horizon=horizon, simulator=simulator,
               protocol='OCBench task-reward rollouts')
    return row, dict(returns=returns, success=success)


def evaluate_cpu(learner, task, candidates, horizon, seed, rng, returns, success):
    env = open_env(task)
    try:
        for i in range(len(returns)):
            state, _ = env.reset(seed=seed + i)
            for _ in range(horizon):
                observation = learner.tensor(state)[None]
                outcomes = learner.flow.sample(observation, rng, candidates)
                goal = elite_mean(outcomes, outcomes[:, -1], learner.config.elites)
                action = learner._action(observation, goal, rng)
                state, reward, terminated, truncated, info = env.step(action)
                returns[i] += reward
                success[i] |= bool(info.get('success', reward > 0))
                if terminated or truncated:
                    break
    finally:
        env.close()


def evaluate_mjwarp(learner, task, candidates, horizon, seed, rng, returns, success):
    import ocbench
    import warp as wp

    if learner.device.type != 'cuda' or not wp.is_cuda_available():
        raise ValueError('OCBench MJWarp evaluation needs CUDA in both PyTorch and Warp')
    episodes = len(returns)
    worlds_per_plan = max(1, 8192 // candidates)
    low, span = learner.action_low, learner.action_span
    with wp.ScopedDevice(str(learner.device)):
        env = ocbench.make(task, nworld=episodes)
        try:
            states, _ = env.reset(seeds=np.arange(seed, seed + episodes, dtype=np.int32))
            if not env.data.qpos.device.is_cuda:
                raise RuntimeError('OCBench MJWarp initialized without CUDA physics')
            done = np.zeros(episodes, dtype=bool)
            for _ in range(horizon):
                active = np.flatnonzero(~done)
                actions = np.zeros(env.action_space.shape, dtype=np.float32)
                for start in range(0, len(active), worlds_per_plan):
                    ids = active[start:start + worlds_per_plan]
                    obs = learner.tensor(states[ids])
                    outcomes = learner.flow.sample(obs.repeat_interleave(candidates, dim=0), rng)
                    outcomes = outcomes.reshape(len(ids), candidates, learner.config.feature_dim)
                    goals = elite_mean(outcomes, outcomes[:, None, :, -1],
                                       learner.config.elites).squeeze(1)
                    samples = learner.readout.sample(torch.cat((obs, goals), dim=1), rng)
                    actions[ids] = (low + (samples + 1) * .5 * span).cpu().numpy()
                states, reward, terminated, truncated, info = env.step(actions)
                returns[active] += np.asarray(reward)[active]
                success[active] |= np.asarray(info.get('success', reward > 0))[active]
                done |= np.asarray(terminated) | np.asarray(truncated)
                if done.all():
                    break
        finally:
            env.close()
