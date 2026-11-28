"""Train/resume the transition-based flow implementation."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
from tqdm import tqdm
from .core import Config, Learner, resolve_device
from .evaluate import DEFAULT_BANKS, DEFAULT_TASKS, evaluate_learner, save_evaluation
from .ocbench_adapter import DEFAULT_TASK as DEFAULT_OCBENCH_TASK
from .ogbench_adapter import DATASETS, EVAL_UPDATES, TRAIN_UPDATES, epoch_batches


def save_round(learner, output, spec, snapshot=False):
    checkpoint = output / 'checkpoint.npz'
    learner.save(checkpoint, spec)
    if snapshot:
        destination = output / f'round{learner.round:03d}.npz'
        temporary = destination.with_suffix('.tmp')
        temporary.unlink(missing_ok=True)
        try:
            os.link(checkpoint, temporary)
        except OSError:
            shutil.copyfile(checkpoint, temporary)
        os.replace(temporary, destination)


WANDB_CHECKPOINT_EVERY = 100


def upload_checkpoint(run, output, current):
    path = output / f'round{current:03d}.npz'
    run.log_artifact(str(path), name=f'checkpoint-{run.id}', type='checkpoint',
                     aliases=['latest', f'round-{current}'])


def evaluate_round(learner, output, spec, final_round, *, skip_completed=False, run=None):
    every = spec['eval_every']
    current = learner.round
    if not every or spec['environment'] == 'continuous':
        return
    ogbench = spec['environment'] == 'ogbench'
    if ogbench:
        if not learner.flow.steps or learner.flow.steps % (every * spec['updates_per_round']):
            return
    elif current != final_round and (current == 0 or current % every):
        return
    destination = output / 'evaluation' / f'round{current:03d}'
    if skip_completed and all((destination / name).is_file() for name in ('summary.json', 'evaluation.npz')):
        return
    banks, tasks, candidates = spec['eval_banks'], spec['eval_tasks'], learner.config.candidates
    if ogbench:
        candidates = 1
    details = (dict(horizon=spec['episode_steps']) if spec['environment'] in ('fourrooms', 'ocbench', 'ogbench') else dict(banks=banks))
    print(json.dumps(dict(event='eval_start', round=current, tasks=tasks, candidates=candidates, **details)), flush=True)
    row, arrays = evaluate_learner(learner, spec, banks=banks, candidates=candidates, tasks=tasks)
    summary = save_evaluation(destination, row, arrays, banks=banks, candidates=candidates, tasks=tasks, config=learner.config)
    print(json.dumps(dict(event='eval_done', **summary)), flush=True)
    if run is not None:
        run.log({'round': current, **{f'eval/{key}': row[key]
                                     for key in ('mean_return', 'mean_regret', 'mean_oracle', 'success_rate')
                                     if key in row}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'validate'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--wandb-project', default=os.environ.get('WANDB_PROJECT', 'dispo-fm'))
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'],
                        default=os.environ.get('WANDB_MODE', 'online'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--offline-data', type=Path, help='Local offline replay dataset; disables online collection')
    parser.add_argument('--environment', choices=['garnet', 'fourrooms', 'continuous', 'ocbench', 'ogbench'], default='garnet')
    parser.add_argument('--ogbench-dataset', choices=DATASETS, default=DATASETS[0])
    parser.add_argument('--ocbench-task', default=DEFAULT_OCBENCH_TASK)
    parser.add_argument('--ocbench-shards', type=int, default=1)
    parser.add_argument('--branching', type=int, choices=[1, 2], default=1)
    parser.add_argument('--features',choices=['state','state-action','rbf'],default='state')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-seed', type=int, help='Model and training RNG seed for a new run (defaults to --seed)')
    parser.add_argument('--method', choices=['coverage', 'uniform'],
                        help='Coverage-weighted or uniform training on Garnet')
    parser.add_argument('--tilt', type=float, help=f'Coverage weighting exponent (default: {Config.tilt}; resume preserves saved value)')
    parser.add_argument('--uniform-mix', type=float, help=f'Uniform fraction of coverage training weights (default: {Config.uniform_mix}; resume preserves saved value)')
    parser.add_argument('--goal-pool-multiplier', type=int, help='Coverage readout: 1 weights losses; N > 1 selects a batch without replacement from N batches of goals')
    parser.add_argument('--coverage-weighting', choices=['linear', 'power'], help='Coverage normalization (default: linear; resume preserves saved value)')
    parser.add_argument('--gram-ema-rate', type=float, help='Gram EMA decay (default: 0.995; resume preserves saved value)')
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--initial-steps', type=int, default=4096)
    parser.add_argument('--pretrain-updates', type=int)
    parser.add_argument('--steps-per-round', type=int, default=1024, help='Environment steps per collection round')
    parser.add_argument('--collect-every', type=int, help='Rounds between online collections (default: 5)')
    parser.add_argument('--updates-per-round', type=int)
    parser.add_argument('--episode-steps', type=int)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--ode-steps', type=int, help=f'ODE steps (default: {Config.ode_steps})')
    parser.add_argument('--ode-solver', choices=['heun', 'euler'], help=f'ODE solver (default: {Config.ode_solver})')
    parser.add_argument('--candidates', type=int, default=1000)
    parser.add_argument('--save-every', type=int, help='Rounds between checkpoint snapshots (default: 5; 0 keeps only initial and final snapshots)')
    parser.add_argument('--eval-every', type=int, help='Rounds between evaluations (OGBench: 1, Garnet: 5; 0 disables)')
    parser.add_argument('--eval-banks', type=int, help=f'Banks per state for Garnet evaluation (default: {DEFAULT_BANKS})')
    parser.add_argument('--eval-tasks', type=int, help='Evaluation reward tasks or rollout episodes')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--tiny', action='store_true', help='Small network for smoke tests only')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    new_ogbench = not args.resume and args.environment == 'ogbench'
    if args.pretrain_updates is None:
        args.pretrain_updates = 0 if new_ogbench else 3000
    if args.updates_per_round is None:
        args.updates_per_round = EVAL_UPDATES if new_ogbench else 128
    if args.train_seed is not None and (args.train_seed < 0 or args.resume):
        parser.error('--train-seed must be nonnegative and is only for new runs')
    if args.method is None:
        args.method = 'uniform' if args.environment in ('ocbench', 'ogbench') else 'coverage'
    if new_ogbench:
        if args.method != 'uniform' or args.features != 'state' or args.offline_data or args.pretrain_updates:
            parser.error('OGBench uses official offline datasets, uniform batches, state features and no separate pretraining')
    if not args.resume and args.environment == 'ocbench':
        if args.method != 'uniform' or args.features != 'state':
            parser.error('OCBench currently requires --method uniform --features state')
        if args.ocbench_shards < 1:
            parser.error('--ocbench-shards must be positive')

    if not args.resume and args.features == 'rbf' and args.environment != 'fourrooms':
        parser.error('--features rbf requires Four Rooms')
    if args.resume and args.offline_data:
        parser.error('Resume restores its replay; use --offline-data for a new run')
    if args.episode_steps is None and args.environment not in ('ocbench', 'ogbench'):
        args.episode_steps = 64
    for key in ('rounds', 'initial_steps', 'pretrain_updates', 'steps_per_round', 'updates_per_round', 'episode_steps'):
        if getattr(args, key) is not None and getattr(args, key) < (0 if key in ('rounds', 'pretrain_updates') else 1):
            parser.error(f'Invalid {key}')
    for key, minimum in (('collect_every', 1), ('eval_every', 0), ('save_every', 0), ('eval_banks', 1), ('eval_tasks', 1)):
        value = getattr(args, key)
        if value is not None and value < minimum:
            parser.error(f'Invalid {key}')
    if args.ode_steps is not None and args.ode_steps < 1:
        parser.error('--ode-steps must be positive')
    if args.tilt is not None and (not np.isfinite(args.tilt) or args.tilt < 0):
        parser.error('--tilt must be finite and nonnegative')
    if args.goal_pool_multiplier is not None and args.goal_pool_multiplier < 1:
        parser.error('--goal-pool-multiplier must be positive')
    if args.uniform_mix is not None and not 0 <= args.uniform_mix <= 1:
        parser.error('--uniform-mix must be between 0 and 1')
    if args.gram_ema_rate is not None and not 0 <= args.gram_ema_rate < 1:
        parser.error('--gram-ema-rate must be in [0,1)')
    if args.command == 'validate':
        Config(6, 12, batch_size=args.batch_size, ode_steps=args.ode_steps or Config.ode_steps, ode_solver=args.ode_solver or Config.ode_solver, candidates=args.candidates)
        print('Transition-flow configuration valid.')
        return
    if args.output is None:
        parser.error('--output required')
    try:
        device = resolve_device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if device.type == 'cuda':
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from environments import FOURROOMS_GOALS, fourrooms_rbf_features, make_experiment
    print(f'DiSPO FM device: {device}', file=sys.stderr)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output/'run.lock'
    with lock.open('x'):
        run = None
        try:
            if args.resume:
                learner, spec = Learner.load(args.resume, device)
                if args.gram_ema_rate is not None:
                    learner.config.gram_ema_rate = args.gram_ema_rate
                if args.goal_pool_multiplier is not None:
                    learner.config.goal_pool_multiplier = args.goal_pool_multiplier
                if args.coverage_weighting is not None:
                    learner.config.coverage_weighting = args.coverage_weighting
                if args.uniform_mix is not None:
                    learner.config.uniform_mix = args.uniform_mix
                if args.tilt is not None:
                    learner.config.tilt = args.tilt
                if args.ode_steps is not None:
                    learner.config.ode_steps = args.ode_steps
                if args.ode_solver is not None:
                    learner.config.ode_solver = args.ode_solver
            else:
                if (args.output/'checkpoint.npz').exists():
                    parser.error('Checkpoint exists; use --resume')
                spec = {k: getattr(args, k) for k in ('environment', 'branching', 'features', 'seed', 'method', 'initial_steps',
                        'pretrain_updates', 'steps_per_round', 'updates_per_round', 'episode_steps')}
                ocbench = args.environment == 'ocbench'
                continuous = args.environment in ('continuous', 'ocbench', 'ogbench')
                total_updates = Config.total_updates
                action_bounds = {}
                if ocbench:
                    from .ocbench_adapter import load_ocbench_replay, open_env

                    env = open_env(args.ocbench_task)
                    try:
                        obs_dim, = env.observation_space.shape
                        spec['episode_steps'] = args.episode_steps or env.spec.max_episode_steps
                        action_bounds = dict(low=tuple(map(float, env.action_space.low)),
                                             high=tuple(map(float, env.action_space.high)))
                    finally:
                        env.close()
                    spec['ocbench_task'] = args.ocbench_task
                    spec['ocbench_shards'] = args.ocbench_shards
                    replay, spec['ocbench_reward_weights'] = load_ocbench_replay(
                        args.offline_data, args.ocbench_task, args.ocbench_shards, device)
                    spec['ocbench_feature_map'] = 'dispo-fourier'
                    spec['offline_data'] = str(args.offline_data.resolve()) if args.offline_data else 'OCBench release'
                    features = replay.arrays['features']
                    fdim, actions = features.shape[1], 2
                    feature_low = tuple(map(float, features.min(axis=0)[:-1])) + (0.,)
                    feature_high = tuple(map(float, features.max(axis=0)[:-1])) + (1.,)
                elif new_ogbench:
                    from .ogbench_adapter import load_replay, open_env

                    dataset = args.ogbench_dataset
                    env = open_env(dataset)
                    try:
                        obs_dim, = env.observation_space.shape
                        spec['episode_steps'] = args.episode_steps or env.spec.max_episode_steps
                        action_bounds = dict(low=tuple(map(float, env.action_space.low)),
                                             high=tuple(map(float, env.action_space.high)))
                    finally:
                        env.close()
                    print(f'Loading OGBench training data and Fourier features: {dataset}', flush=True)
                    replay = load_replay(dataset, device)
                    spec.update(ogbench_dataset=dataset, offline_data='OGBench release')
                    steps_per_epoch = replay.size // args.batch_size
                    if not 0 < steps_per_epoch <= TRAIN_UPDATES:
                        parser.error('Batch size must yield between 1 and 100000 updates per epoch')
                    total_updates = TRAIN_UPDATES // steps_per_epoch * steps_per_epoch
                    spec.update(steps_per_epoch=steps_per_epoch, default_updates=total_updates)
                    features = replay.arrays['features']
                    fdim, actions = features.shape[1], 2
                    feature_low = tuple(map(float, features.min(axis=0)[:-1])) + (0.,)
                    feature_high = tuple(map(float, features.max(axis=0)[:-1])) + (1.,)
                else:
                    experiment = make_experiment(args.environment, args.seed, teacher_count=0, branching=args.branching)
                    obs_dim = experiment['anchors'].shape[1]
                    fdim = obs_dim if args.features in ('state', 'rbf') else experiment['feature_dim']
                    actions = 2 if continuous else experiment['space'].n
                    feature_low = ((-1., 0.) if args.features == 'state' else (-1., -1., 0., 0.)) if continuous else None
                    feature_high = (1.,) * fdim if continuous else None
                c = Config(obs_dim, fdim, actions=actions,
                           tilt=Config.tilt if args.tilt is None else args.tilt,
                           uniform_mix=Config.uniform_mix if args.uniform_mix is None else args.uniform_mix,
                           goal_pool_multiplier=args.goal_pool_multiplier or 1,
                           coverage_weighting=args.coverage_weighting or Config.coverage_weighting,
                           gram_ema_rate=Config.gram_ema_rate if args.gram_ema_rate is None else args.gram_ema_rate,
                           outcome_simplex=args.environment == 'garnet',
                           outcome_step_clip=args.environment == 'fourrooms',
                           coverage_state_rbf=args.environment == 'fourrooms',
                           action_kind='continuous' if continuous else 'discrete',
                           seed=args.seed if args.train_seed is None else args.train_seed,
                           batch_size=args.batch_size, ode_steps=args.ode_steps or Config.ode_steps, ode_solver=args.ode_solver or Config.ode_solver, candidates=args.candidates,
                           hidden=8 if args.tiny else 256,down_dims=(8,16,32) if args.tiny else (256,512,1024),
                           feature_low=feature_low, feature_high=feature_high, total_updates=total_updates, **action_bounds)
                learner = Learner(c, device)
                if ocbench or new_ogbench:
                    learner.replay = replay
                elif args.offline_data:
                    from .offline_data import load_replay
                    learner.replay = load_replay(args.offline_data, spec)
                    spec['offline_data'] = str(args.offline_data.resolve())
            ogbench = spec['environment'] == 'ogbench'
            if ogbench:
                updates_per_round = spec['updates_per_round']
                if args.rounds is None:
                    target_updates = spec['default_updates']
                    args.rounds = (target_updates + updates_per_round - 1) // updates_per_round
                else:
                    target_updates = args.rounds * updates_per_round
                batches = epoch_batches(learner.replay.size, learner.config.batch_size,
                                        learner.config.seed, learner.flow.steps)
            elif args.rounds is None:
                args.rounds = 100
            if args.rounds < learner.round or (ogbench and target_updates < learner.flow.steps):
                parser.error('--rounds is below checkpoint round')
            collect_every = args.collect_every if args.collect_every is not None else spec.get('collect_every', 5)
            spec['collect_every'] = collect_every
            save_every = args.save_every if args.save_every is not None else spec.get('save_every', 0 if ogbench else 5)
            spec['save_every'] = save_every
            evaluation_defaults = (('eval_every', 1 if ogbench else 5), ('eval_banks', DEFAULT_BANKS),
                                   ('eval_tasks', 1 if ogbench else 10 if spec['environment'] == 'ocbench' else DEFAULT_TASKS))
            evaluation_overridden = any(getattr(args, key) is not None for key, _ in evaluation_defaults)
            for key, default in evaluation_defaults:
                value = getattr(args, key)
                spec[key] = spec.get(key, default) if value is None else value
            if spec['environment'] not in ('ocbench', 'ogbench'):
                if args.resume:
                    experiment = make_experiment(spec['environment'], spec['seed'], teacher_count=0, branching=spec['branching'])
                env = experiment['env_factory']()
                feature = experiment['feature']
                if spec.get('features','state-action')=='state':
                    feature=(lambda obs,action:np.array([obs[0],obs[0]**2])) if spec['environment']=='continuous' else (lambda obs,action:np.asarray(obs).copy())
                if spec.get('features') == 'rbf':
                    rbf = fourrooms_rbf_features()
                    feature = lambda obs, action: rbf[obs.argmax()]
                    spec['eval_tasks'] = len(FOURROOMS_GOALS)
            coverage = spec['environment'] in ('garnet', 'fourrooms') and spec['method'] == 'coverage'
            if args.wandb_mode != 'disabled':
                import wandb

                output_path = str(args.output.resolve())
                previous = spec.get('wandb', {})
                resume_run = previous.get('output') == output_path and previous.get('project') == args.wandb_project
                run = wandb.init(
                    project=args.wandb_project, name=args.output.name, mode=args.wandb_mode,
                    id=previous['id'] if resume_run else None,
                    resume='allow' if resume_run and args.wandb_mode == 'online' else None,
                    config={'training': asdict(learner.config),
                            'experiment': {key: value for key, value in spec.items() if key != 'wandb'}},
                )
                spec['wandb'] = {'id': run.id, 'project': args.wandb_project, 'output': output_path}
                run.define_metric('round')
                run.define_metric('train/*', step_metric='round')
                run.define_metric('eval/*', step_metric='round')
            if not args.resume:
                if not spec.get('offline_data'):
                    learner.collect(env, feature, spec['initial_steps'], warmup=True, episode_steps=spec['episode_steps'])
                updates = spec['pretrain_updates']
                for _ in tqdm(range(updates), desc='Pretraining', unit='update',
                              dynamic_ncols=True, disable=updates == 0):
                    learner.update_tensor(coverage=coverage)
                save_round(learner, args.output, spec, snapshot=True)
                if run is not None:
                    upload_checkpoint(run, args.output, learner.round)
            resuming_latest = args.resume and args.resume.resolve() == (args.output / 'checkpoint.npz').resolve()
            evaluate_round(learner, args.output, spec, args.rounds,
                           skip_completed=resuming_latest and not evaluation_overridden, run=run)
            while (learner.flow.steps < target_updates if ogbench else learner.round < args.rounds):
                next_round = learner.round + 1
                collected_steps = 0
                if not spec.get('offline_data') and next_round % collect_every == 0:
                    collected_steps = spec['steps_per_round']
                    learner.collect(env, feature, collected_steps, method='uniform', episode_steps=spec['episode_steps'])
                if ogbench:
                    updates = min(updates_per_round - learner.flow.steps % updates_per_round,
                                  target_updates - learner.flow.steps)
                    losses = []
                    with tqdm(total=updates, desc=f'OGBench updates from {learner.flow.steps}', unit='update') as progress:
                        for _ in range(updates):
                            losses.append(learner.update_tensor(indices=next(batches)))
                            progress.update()
                    learner.round = (learner.flow.steps + updates_per_round - 1) // updates_per_round
                else:
                    losses = [learner.update_tensor(coverage=coverage) for _ in range(spec['updates_per_round'])]
                    learner.round += 1
                flow_loss, readout_loss = torch.stack(losses).mean(dim=0).tolist()
                row = dict(round=learner.round, transitions=learner.replay.size, env_steps=learner.env_steps,
                           collected_steps=collected_steps, flow_loss=flow_loss, readout_loss=readout_loss)
                if ogbench:
                    row['updates'] = learner.flow.steps
                learner.history.append(row)
                snapshot = learner.round == args.rounds or (save_every > 0 and learner.round % save_every == 0)
                publish = run is not None and (
                    ogbench or learner.round % WANDB_CHECKPOINT_EVERY == 0)
                save_round(learner, args.output, spec, snapshot=snapshot or publish)
                print(json.dumps(row), flush=True)
                if run is not None:
                    run.log({'round': learner.round, **{f'train/{key}': value
                                                        for key, value in row.items() if key != 'round'}})
                    if publish:
                        upload_checkpoint(run, args.output, learner.round)
                evaluate_round(learner, args.output, spec, args.rounds, run=run)
        finally:
            lock.unlink()
            if run is not None:
                run.finish()


if __name__ == '__main__':
    main()
