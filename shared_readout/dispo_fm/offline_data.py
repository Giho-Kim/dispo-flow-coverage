"""Build balanced one-step transitions for discrete environments using the simulator."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from .core import Replay


DATASET_SCHEMAS = {'garnet': 'garnet-offline-v1', 'fourrooms': 'fourrooms-offline-v1'}


def make_discrete_experiment(environment, seed, branching):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from environments import make_experiment

    return make_experiment(environment, seed, teacher_count=0, branching=branching)


def load_replay(path, spec):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        expected_schema = DATASET_SCHEMAS.get(spec['environment'])
        if expected_schema is None or metadata.get('schema') != expected_schema:
            raise ValueError('Offline dataset schema does not match the environment')
        for key in ('environment', 'seed', 'branching', 'features'):
            if metadata[key] != spec[key]:
                raise ValueError(f'Offline dataset does not match run setting {key}')
        arrays = {key: data[f'replay/{key}'] for key in Replay.fields}
    size = metadata['transitions']
    experiment = make_discrete_experiment(spec['environment'], spec['seed'], spec['branching'])
    obs_dim = experiment['anchors'].shape[1]
    action_count = experiment['space'].n
    feature_dim = obs_dim if spec['features'] in ('state', 'rbf') else experiment['feature_dim']
    shapes = dict(observations=(size, obs_dim), next_observations=(size, obs_dim),
                  features=(size, feature_dim), actions=(size,), terminated=(size,), truncated=(size,))
    if size < 1 or any(value.shape != shapes[key] or not np.isfinite(value).all()
                       for key, value in arrays.items()):
        raise ValueError('Invalid offline replay arrays')
    actions = arrays['actions']
    if not np.issubdtype(actions.dtype, np.integer) or np.any((actions < 0) | (actions >= action_count)):
        raise ValueError('Invalid offline actions')
    if arrays['terminated'].any():
        raise ValueError('Offline dataset must be nonterminating')
    replay = Replay()
    replay.arrays = arrays
    replay.size = replay.capacity = size
    return replay


def build(output, samples_per_pair=512, seed=0, branching=1, features='state', *, environment='garnet', samples_per_pair_range=None, count_seed=2026):
    if samples_per_pair < 1:
        raise ValueError('samples_per_pair must be positive')
    if features not in ('state', 'state-action', 'rbf'):
        raise ValueError('Expected state, state-action or rbf features')
    if environment not in DATASET_SCHEMAS:
        raise ValueError('Expected Garnet or Four Rooms')
    experiment = make_discrete_experiment(environment, seed, branching)
    env = experiment['env_factory']()
    feature_fn = experiment['feature']
    if features == 'rbf':
        if environment != 'fourrooms':
            raise ValueError('RBF features require Four Rooms')
        from environments import fourrooms_rbf_features

        rbf = fourrooms_rbf_features()
        feature_fn = lambda obs, action: rbf[obs.argmax()]
    states, actions = len(experiment['contexts']), experiment['space'].n
    rng = np.random.default_rng(seed)
    if samples_per_pair_range is None:
        desired_counts = np.full((states, actions), samples_per_pair, dtype=int)
    else:
        low, high = samples_per_pair_range
        if low < 1 or high < low:
            raise ValueError('Invalid samples-per-pair range')
        desired_counts = np.random.default_rng(count_seed).integers(low, high + 1, size=(states, actions))
    replay = Replay()
    counts = np.zeros((states, actions), dtype=int)
    next_counts = np.zeros((states, actions, states), dtype=int)
    for state in experiment['contexts']:
        for action in range(actions):
            for _ in range(desired_counts[state, action]):
                observation = env.reset_to(state, seed=int(rng.integers(2**31)))
                next_observation, _, terminated, truncated, _ = env.step(action)
                feature = (observation.copy() if features == 'state'
                           else feature_fn(observation, action))
                replay.add(observation, action, feature, next_observation, terminated, truncated)
                counts[state, action] += 1
                next_counts[state, action, next_observation.argmax()] += 1
    if not np.array_equal(counts, desired_counts):
        raise RuntimeError('State-action counts do not match requested counts')
    if (environment == 'fourrooms' or branching == 1) and not np.all((next_counts > 0).sum(axis=2) == 1):
        raise RuntimeError('Inconsistent deterministic transitions')
    metadata = dict(schema=DATASET_SCHEMAS[environment], environment=environment, seed=seed,
                    branching=branching, features=features, states=states, actions=actions,
                    samples_per_pair=samples_per_pair, transitions=replay.size,
                    counts=counts.tolist(), next_state_counts=next_counts.tolist(),
                    collection='independent one-step transitions, reset to each source state')
    if samples_per_pair_range is not None:
        metadata.pop('samples_per_pair')
        metadata['samples_per_pair_range'] = list(samples_per_pair_range)
        metadata['sampling_seed'] = count_seed
    # Shuffle the file so consecutive minibatches also mix source states and actions.
    order = rng.permutation(replay.size)
    arrays = {f'replay/{key}': value[order] for key, value in replay.arrays.items()}
    arrays['metadata'] = np.array(json.dumps(metadata))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--environment', choices=DATASET_SCHEMAS, default='garnet')
    parser.add_argument('--samples-per-pair', type=int, default=512)
    parser.add_argument('--samples-per-pair-range', type=int, nargs=2, metavar=('MIN', 'MAX'))
    parser.add_argument('--count-seed', type=int, default=2026)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--branching', type=int, choices=[1, 2], default=1)
    parser.add_argument('--features', choices=['state', 'state-action', 'rbf'], default='state')
    args = parser.parse_args()
    if args.samples_per_pair < 1:
        parser.error('--samples-per-pair must be positive')
    if args.samples_per_pair_range and (args.samples_per_pair_range[0] < 1 or args.samples_per_pair_range[1] < args.samples_per_pair_range[0]):
        parser.error('Invalid samples-per-pair range')
    metadata = build(args.output, args.samples_per_pair, args.seed, args.branching, args.features,
                     environment=args.environment, samples_per_pair_range=args.samples_per_pair_range,
                     count_seed=args.count_seed)
    print(json.dumps({key: value for key, value in metadata.items()
                      if key not in ('counts', 'next_state_counts')}, indent=2))


if __name__ == '__main__':
    main()
