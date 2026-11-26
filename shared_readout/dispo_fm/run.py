"""Train/resume the transition-based flow implementation."""
import argparse
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


def evaluate_round(learner, output, spec, final_round, *, skip_completed=False):
    every = spec['eval_every']
    current = learner.round
    if not every or spec['environment'] == 'continuous':
        return
    if current != final_round and (current == 0 or current % every):
        return
    destination = output / 'evaluation' / f'round{current:03d}'
    if skip_completed and all((destination / name).is_file() for name in ('summary.json', 'evaluation.npz')):
        return
    banks, tasks, candidates = spec['eval_banks'], spec['eval_tasks'], learner.config.candidates
    details = (dict(horizon=spec['episode_steps']) if spec['environment'] == 'fourrooms' else dict(banks=banks))
    print(json.dumps(dict(event='eval_start', round=current, tasks=tasks, candidates=candidates, **details)), flush=True)
    row, arrays = evaluate_learner(learner, spec, banks=banks, candidates=candidates, tasks=tasks)
    summary = save_evaluation(destination, row, arrays, banks=banks, candidates=candidates, tasks=tasks, config=learner.config)
    print(json.dumps(dict(event='eval_done', **summary)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'validate'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--offline-data', type=Path, help='Fixed discrete replay dataset; disables online collection')
    parser.add_argument('--environment', choices=['garnet', 'fourrooms', 'continuous'], default='garnet')
    parser.add_argument('--branching', type=int, choices=[1, 2], default=1)
    parser.add_argument('--features',choices=['state','state-action','rbf'],default='state')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', choices=['coverage', 'uniform'], default='coverage',
                        help='Coverage-weighted or uniform training on Garnet')
    parser.add_argument('--tilt', type=float, help=f'Coverage weighting exponent (default: {Config.tilt}; resume preserves saved value)')
    parser.add_argument('--uniform-mix', type=float, help=f'Uniform fraction of coverage training weights (default: {Config.uniform_mix}; resume preserves saved value)')
    parser.add_argument('--goal-pool-multiplier', type=int, help='Coverage readout: 1 weights losses; N > 1 selects a batch without replacement from N batches of goals')
    parser.add_argument('--coverage-weighting', choices=['linear', 'power'], help='Coverage normalization (default: linear; resume preserves saved value)')
    parser.add_argument('--gram-ema-rate', type=float, help='Gram EMA decay (default: 0.995; resume preserves saved value)')
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--initial-steps', type=int, default=4096)
    parser.add_argument('--pretrain-updates', type=int, default=3000)
    parser.add_argument('--steps-per-round', type=int, default=1024, help='Environment steps per collection round')
    parser.add_argument('--collect-every', type=int, help='Rounds between online collections (default: 5)')
    parser.add_argument('--updates-per-round', type=int, default=128)
    parser.add_argument('--episode-steps', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--ode-steps', type=int, help=f'ODE steps (default: {Config.ode_steps})')
    parser.add_argument('--ode-solver', choices=['heun', 'euler'], help=f'ODE solver (default: {Config.ode_solver})')
    parser.add_argument('--candidates', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, help='Garnet rounds between evaluations (default: 5, 0 disables)')
    parser.add_argument('--eval-banks', type=int, help=f'Banks per state for Garnet evaluation (default: {DEFAULT_BANKS})')
    parser.add_argument('--eval-tasks', type=int, help=f'Garnet evaluation reward tasks (default: {DEFAULT_TASKS})')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--tiny', action='store_true', help='Small network for smoke tests only')
    args = parser.parse_args()
    if not args.resume and args.features == 'rbf' and args.environment != 'fourrooms':
        parser.error('--features rbf requires Four Rooms')
    if args.resume and args.offline_data:
        parser.error('Resume restores its replay; use --offline-data for a new run')
    for key in ('rounds', 'initial_steps', 'pretrain_updates', 'steps_per_round', 'updates_per_round', 'episode_steps'):
        if getattr(args, key) < (0 if key in ('rounds', 'pretrain_updates') else 1):
            parser.error(f'Invalid {key}')
    for key, minimum in (('collect_every', 1), ('eval_every', 0), ('eval_banks', 1), ('eval_tasks', 1)):
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
                continuous = args.environment == 'continuous'
                experiment = make_experiment(args.environment, args.seed, teacher_count=0, branching=args.branching)
                obs_dim = experiment['anchors'].shape[1]
                fdim = obs_dim if args.features in ('state', 'rbf') else experiment['feature_dim']
                actions = 2 if continuous else experiment['space'].n
                c = Config(obs_dim, fdim, actions=actions,
                           tilt=Config.tilt if args.tilt is None else args.tilt,
                           uniform_mix=Config.uniform_mix if args.uniform_mix is None else args.uniform_mix,
                           goal_pool_multiplier=args.goal_pool_multiplier or 1,
                           coverage_weighting=args.coverage_weighting or Config.coverage_weighting,
                           gram_ema_rate=Config.gram_ema_rate if args.gram_ema_rate is None else args.gram_ema_rate,
                           outcome_simplex=args.environment == 'garnet',
                           outcome_step_clip=args.environment == 'fourrooms',
                           coverage_state_rbf=args.environment == 'fourrooms',
                           action_kind='continuous' if continuous else 'discrete', seed=args.seed,
                           batch_size=args.batch_size, ode_steps=args.ode_steps or Config.ode_steps, ode_solver=args.ode_solver or Config.ode_solver, candidates=args.candidates,
                           hidden=8 if args.tiny else 256,down_dims=(8,16,32) if args.tiny else (256,512,1024),
                           feature_low=((-1.,0.) if args.features=='state' else (-1.,-1.,0.,0.)) if continuous else None,
                           feature_high=((1.,1.) if args.features=='state' else (1.,1.,1.,1.)) if continuous else None)
                learner = Learner(c, device)
                if args.offline_data:
                    from .offline_data import load_replay
                    learner.replay = load_replay(args.offline_data, spec)
                    spec['offline_data'] = str(args.offline_data.resolve())
            if args.rounds < learner.round:
                parser.error('--rounds is below checkpoint round')
            collect_every = args.collect_every if args.collect_every is not None else spec.get('collect_every', 5)
            spec['collect_every'] = collect_every
            evaluation_defaults = (('eval_every', 5), ('eval_banks', DEFAULT_BANKS), ('eval_tasks', DEFAULT_TASKS))
            evaluation_overridden = any(getattr(args, key) is not None for key, _ in evaluation_defaults)
            for key, default in evaluation_defaults:
                value = getattr(args, key)
                spec[key] = spec.get(key, default) if value is None else value
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
            coverage = spec['environment'] != 'continuous' and spec['method'] == 'coverage'
            if not args.resume:
                if not spec.get('offline_data'):
                    learner.collect(env, feature, spec['initial_steps'], warmup=True, episode_steps=spec['episode_steps'])
                updates = spec['pretrain_updates']
                for _ in tqdm(range(updates), desc='Pretraining', unit='update',
                              dynamic_ncols=True, disable=updates == 0):
                    learner.update_tensor(coverage=coverage)
                save_round(learner, args.output, spec, snapshot=True)
            resuming_latest = args.resume and args.resume.resolve() == (args.output / 'checkpoint.npz').resolve()
            evaluate_round(learner, args.output, spec, args.rounds,
                           skip_completed=resuming_latest and not evaluation_overridden)
            while learner.round < args.rounds:
                next_round = learner.round + 1
                collected_steps = 0
                if not spec.get('offline_data') and next_round % collect_every == 0:
                    collected_steps = spec['steps_per_round']
                    learner.collect(env, feature, collected_steps, method='uniform', episode_steps=spec['episode_steps'])
                losses = [learner.update_tensor(coverage=coverage) for _ in range(spec['updates_per_round'])]
                flow_loss, readout_loss = torch.stack(losses).mean(dim=0).tolist()
                learner.round += 1
                row = dict(round=learner.round, transitions=learner.replay.size, env_steps=learner.env_steps,
                           collected_steps=collected_steps, flow_loss=flow_loss, readout_loss=readout_loss)
                learner.history.append(row)
                save_round(learner, args.output, spec, snapshot=learner.round % 5 == 0 or learner.round == args.rounds)
                print(json.dumps(row), flush=True)
                evaluate_round(learner, args.output, spec, args.rounds)
        finally:
            lock.unlink()


if __name__ == '__main__':
    main()
