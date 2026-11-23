"""Train/resume the transition-based flow implementation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from .core import Config, Learner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'validate'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--environment', choices=['garnet', 'continuous'], default='garnet')
    parser.add_argument('--branching', type=int, choices=[1, 2], default=1)
    parser.add_argument('--features',choices=['state','state-action'],default='state')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', choices=['coverage', 'uniform'], default='coverage')
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--initial-steps', type=int, default=4096)
    parser.add_argument('--pretrain-updates', type=int, default=3000)
    parser.add_argument('--steps-per-round', type=int, default=1024)
    parser.add_argument('--updates-per-round', type=int, default=128)
    parser.add_argument('--episode-steps', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--ode-steps', type=int, default=24)
    parser.add_argument('--candidates', type=int, default=1000)
    parser.add_argument('--tiny', action='store_true', help='Small network for smoke tests only')
    args = parser.parse_args()
    for key in ('rounds', 'initial_steps', 'pretrain_updates', 'steps_per_round', 'updates_per_round', 'episode_steps'):
        if getattr(args, key) < (0 if key in ('rounds', 'pretrain_updates') else 1):
            parser.error(f'Invalid {key}')
    if args.command == 'validate':
        Config(6, 12, batch_size=args.batch_size, ode_steps=args.ode_steps, candidates=args.candidates)
        print('Transition-flow configuration valid.')
        return
    if args.output is None:
        parser.error('--output required')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output/'run.lock'
    with lock.open('x'):
        try:
            if args.resume:
                learner, spec = Learner.load(args.resume)
            else:
                if (args.output/'checkpoint.npz').exists():
                    parser.error('Checkpoint exists; use --resume')
                spec = {k: getattr(args, k) for k in ('environment', 'branching', 'features', 'seed', 'method', 'initial_steps',
                        'pretrain_updates', 'steps_per_round', 'updates_per_round', 'episode_steps')}
                continuous = args.environment == 'continuous'
                fdim=(2 if continuous else 6) if args.features=='state' else (4 if continuous else 12)
                c = Config(2 if continuous else 6, fdim,
                           action_kind='continuous' if continuous else 'discrete', seed=args.seed,
                           batch_size=args.batch_size, ode_steps=args.ode_steps, candidates=args.candidates,
                           hidden=8 if args.tiny else 256,down_dims=(8,16,32) if args.tiny else (256,512,1024),
                           feature_low=((-1.,0.) if args.features=='state' else (-1.,-1.,0.,0.)) if continuous else None,
                           feature_high=((1.,1.) if args.features=='state' else (1.,1.,1.,1.)) if continuous else None)
                learner = Learner(c)
            # Import only simulator adapters; no old training/replay modules.
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from environments import make_experiment
            experiment = make_experiment(spec['environment'], spec['seed'], teacher_count=0, branching=spec['branching'])
            env = experiment['env_factory']()
            feature = experiment['feature']
            if spec.get('features','state-action')=='state':
                feature=(lambda obs,action:np.array([obs[0],obs[0]**2])) if spec['environment']=='continuous' else (lambda obs,action:np.asarray(obs).copy())
            if not args.resume:
                learner.collect(env, feature, spec['initial_steps'], warmup=True, episode_steps=spec['episode_steps'])
                for _ in range(spec['pretrain_updates']):
                    learner.update()
                learner.save(args.output/'checkpoint.npz', spec)
                learner.save(args.output/'round000.npz', spec)
            if args.rounds < learner.round:
                parser.error('--rounds is below checkpoint round')
            while learner.round < args.rounds:
                learner.collect(env, feature, spec['steps_per_round'], method=spec['method'], episode_steps=spec['episode_steps'])
                losses = [learner.update() for _ in range(spec['updates_per_round'])]
                learner.round += 1
                row = dict(round=learner.round, transitions=learner.replay.size, env_steps=learner.env_steps,
                           **{k: float(np.mean([x[k] for x in losses])) for k in losses[0]})
                learner.history.append(row)
                learner.save(args.output/'checkpoint.npz', spec)
                if learner.round % 5 == 0 or learner.round == args.rounds:
                    learner.save(args.output/f'round{learner.round:03d}.npz', spec)
                print(json.dumps(row), flush=True)
        finally:
            lock.unlink()


if __name__ == '__main__':
    main()
