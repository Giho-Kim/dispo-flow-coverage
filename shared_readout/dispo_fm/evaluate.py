"""Fresh candidate-bank inference, integrating sampling into action probabilities."""
from pathlib import Path
import itertools
import json
import sys
import numpy as np
from .core import Learner


def evaluate(path, banks=128, candidates=None, tasks=128, seed=93281):
    learner,spec=Learner.load(path)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from environments import make_experiment
    p=make_experiment('garnet',spec['seed'],0,spec['branching'])['env_factory']()._p
    states,actions=p.shape[:2]; gamma=learner.config.gamma
    candidates=learner.config.candidates if candidates is None else candidates
    rng=np.random.default_rng(seed)
    rewards=rng.normal(size=(tasks,learner.config.feature_dim));rewards/=np.linalg.norm(rewards,axis=1,keepdims=True)
    reward_sa=np.repeat(rewards[:,:,None],actions,axis=2) if spec.get('features','state-action')=='state' else rewards.reshape(tasks,states,actions)
    policy=np.zeros((tasks,states,actions))
    for s in range(states):
        rng=np.random.default_rng(seed+10000*s)
        # Each bank is an independent draw for one visit. Average readout probabilities,
        # then solve the Markov reward process of the fresh-resampling policy.
        for start in range(0,banks,8):
            n=min(8,banks-start)
            obs=np.repeat(np.eye(states)[s][None],n*candidates,axis=0)
            goals=learner.target.sample(obs,rng,learner.config.ode_steps).reshape(n,candidates,-1)
            scores=np.einsum('td,bcd->btc',rewards,goals)
            elite=np.argsort(-scores,axis=2)[:,:,:learner.config.elites]
            selected=goals[np.arange(n)[:,None,None],elite].mean(2).reshape(n*tasks,-1)
            context=np.concatenate([np.repeat(np.eye(states)[s][None],n*tasks,axis=0),selected],axis=1)
            action_vectors=learner.readout_target.sample(context,rng,learner.config.ode_steps)
            chosen=action_vectors.argmax(1).reshape(n,tasks)
            policy[:,s]+=np.eye(actions)[chosen].sum(0)/banks
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


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--banks',type=int,default=128)
    parser.add_argument('--candidates',type=int,default=None)
    parser.add_argument('--tasks',type=int,default=128)
    args=parser.parse_args()
    learner,_=Learner.load(args.checkpoint)
    row,arrays=evaluate(args.checkpoint,banks=args.banks,candidates=args.candidates or learner.config.candidates,tasks=args.tasks)
    row.update(banks=args.banks,candidates=args.candidates or learner.config.candidates,tasks=args.tasks,protocol='fresh outcome and action generation at every visit; action probabilities estimated by independent draws')
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'evaluation.npz',**arrays)
    (args.output/'summary.json').write_text(json.dumps(row,indent=2))
    print(json.dumps(row,indent=2))

if __name__=='__main__':main()
