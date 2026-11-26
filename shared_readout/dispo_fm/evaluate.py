"""Fresh candidate-bank inference, integrating sampling into action probabilities."""
from pathlib import Path
import itertools
import json
import os
import sys
import numpy as np
import torch
from .core import Learner, elite_mean, resolve_device

DEFAULT_BANKS = 128
DEFAULT_TASKS = 128


def evaluate(path, banks=DEFAULT_BANKS, candidates=None, tasks=DEFAULT_TASKS, seed=93281, device='cpu'):
    learner, spec = Learner.load(path, device)
    return evaluate_learner(learner, spec, banks, candidates, tasks, seed)


@torch.no_grad()
def evaluate_learner(learner, spec, banks=DEFAULT_BANKS, candidates=None, tasks=DEFAULT_TASKS, seed=93281):
    if spec is None or spec['environment'] not in ('garnet', 'fourrooms'):
        raise ValueError('The Markov reward process evaluator requires a Garnet or Four Rooms checkpoint')
    config = learner.config
    candidates = config.candidates if candidates is None else candidates
    if min(banks, tasks) < 1 or candidates < config.elites:
        raise ValueError('Expected positive banks/tasks and at least as many candidates as elites')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from environments import make_experiment
    p = make_experiment(spec['environment'], spec['seed'], 0, spec['branching'])['env_factory']()._p
    states, actions = p.shape[:2]
    gamma = config.gamma
    rng = np.random.default_rng(seed)
    if spec.get('features') == 'rbf':
        from environments import FOURROOMS_GOALS, fourrooms_cells, fourrooms_rbf_features

        cells = fourrooms_cells()
        tasks = len(FOURROOMS_GOALS)
        rewards = np.eye(config.feature_dim)[[cells.index(goal) for goal in FOURROOMS_GOALS]]
        reward_sa = np.repeat((rewards @ fourrooms_rbf_features().T)[:, :, None], actions, axis=2)
    else:
        rewards = rng.normal(size=(tasks, config.feature_dim))
        rewards /= np.linalg.norm(rewards, axis=1, keepdims=True)
        reward_sa = (np.repeat(rewards[:, :, None], actions, axis=2) if spec.get('features', 'state-action') == 'state'
                     else rewards.reshape(tasks, states, actions))
    if spec['environment'] == 'fourrooms':
        # One fresh plan and one sampled action per visited state, as in DiSPO.
        horizon = spec.get('episode_steps', 64)
        next_states = p.argmax(axis=2)
        reward_tensor = learner.tensor(rewards)
        observations = torch.eye(states, dtype=learner.dtype, device=learner.device)
        model_rng = torch.Generator(device=learner.device).manual_seed(seed)
        returns = np.zeros(tasks)
        for task in range(tasks):
            state = 0
            for step in range(horizon):
                observation = observations[state:state + 1]
                goals = learner.flow.sample(observation, model_rng, candidates)
                goal = elite_mean(goals, goals @ reward_tensor[task], config.elites)
                action = learner._action(observation, goal.reshape(-1), model_rng)
                returns[task] += gamma ** step * reward_sa[task, state, action]
                state = next_states[state, action]
        optimal = np.zeros((tasks, states))
        for _ in range(horizon):
            optimal = (reward_sa + gamma * optimal[:, next_states]).max(axis=2)
        scale = sum(gamma ** step for step in range(horizon))
        returns /= scale
        oracle = optimal[:, 0] / scale
        row = dict(round=learner.round, mean_return=float(returns.mean()),
                   mean_regret=float((oracle - returns).mean()), mean_oracle=float(oracle.mean()),
                   horizon=horizon, protocol='DiSPO rollout with fresh planning at each step')
        if spec.get('features') == 'rbf':
            row['goal_cells'] = FOURROOMS_GOALS
        return row, dict(returns=returns, oracle=oracle, rewards=rewards)

    reward_tensor = learner.tensor(rewards)
    observations = torch.eye(states, dtype=learner.dtype, device=learner.device)
    counts = torch.zeros((tasks, states, actions), dtype=torch.int64, device=learner.device)
    for s, observation in enumerate(observations.split(1)):
        model_rng = torch.Generator(device=learner.device).manual_seed(seed + 10000 * s)
        for start in range(0, banks, 8):
            n = min(8, banks - start)
            goals = learner.flow.sample(observation, model_rng, n * candidates).reshape(n, candidates, -1)
            scores = reward_tensor @ goals.transpose(1, 2)
            selected = elite_mean(goals, scores, config.elites).reshape(n * tasks, -1)
            context = torch.cat((observation.expand(n * tasks, -1), selected), dim=1)
            chosen = learner.readout.sample(context, model_rng).argmax(dim=1).reshape(n, tasks)
            counts[:, s] += (chosen[..., None] == torch.arange(actions, device=learner.device)).sum(dim=0)
    policy = counts.cpu().numpy() / banks
    kernel=np.einsum('tsa,san->tsn',policy,p)
    immediate=(policy*reward_sa).sum(2)
    values=np.linalg.solve(np.eye(states)[None]-gamma*kernel,immediate[...,None])[...,0]
    returns=(1-gamma)*values[:,0]
    oracle=np.full(tasks,-np.inf)
    for table in itertools.product(range(actions),repeat=states):
        ids=np.array(table)
        r=reward_sa[:,np.arange(states),ids]
        v=np.linalg.solve(np.eye(states)-gamma*p[np.arange(states),ids],r.T).T
        oracle=np.maximum(oracle,(1-gamma)*v[:,0])
    return dict(round=learner.round,mean_return=float(returns.mean()),mean_regret=float((oracle-returns).mean()),mean_oracle=float(oracle.mean())),dict(returns=returns,oracle=oracle,policy=policy,rewards=rewards)


def save_evaluation(output, row, arrays, *, banks, candidates, tasks, config):
    summary = dict(row, candidates=candidates, tasks=len(arrays['returns']),
                   ode_solver=config.ode_solver, ode_steps=config.ode_steps,
                   outcome_simplex=config.outcome_simplex,
                   protocol=row.get('protocol', 'fresh outcome and action generation at every visit; action probabilities estimated by independent draws'))
    if 'horizon' not in row:
        summary['banks'] = banks
    output.mkdir(parents=True, exist_ok=True)
    arrays_temporary = output / 'evaluation.tmp.npz'
    np.savez_compressed(arrays_temporary, **arrays)
    os.replace(arrays_temporary, output / 'evaluation.npz')
    temporary = output / 'summary.tmp'
    temporary.write_text(json.dumps(summary, indent=2))
    os.replace(temporary, output / 'summary.json')
    return summary


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--banks',type=int,default=DEFAULT_BANKS)
    parser.add_argument('--candidates',type=int,default=None)
    parser.add_argument('--tasks',type=int,default=DEFAULT_TASKS)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--ode-steps', type=int)
    parser.add_argument('--ode-solver', choices=['heun', 'euler'])
    args = parser.parse_args()
    if args.ode_steps is not None and args.ode_steps < 1:
        parser.error('--ode-steps must be positive')
    try:
        device = resolve_device(args.device)
        if device.type == 'cuda':
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
        learner, spec = Learner.load(args.checkpoint, device)
        if args.ode_steps is not None:
            learner.config.ode_steps = args.ode_steps
        if args.ode_solver is not None:
            learner.config.ode_solver = args.ode_solver
        candidates = learner.config.candidates if args.candidates is None else args.candidates
        row, arrays = evaluate_learner(learner, spec, args.banks, candidates, args.tasks)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    summary = save_evaluation(args.output, row, arrays, banks=args.banks,
                              candidates=candidates, tasks=args.tasks, config=learner.config)
    print(json.dumps(summary, indent=2))

if __name__=='__main__':main()
