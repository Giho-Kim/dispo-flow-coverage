"""Transition-based DiSPO updates with conditional flow matching (NumPy)."""
from dataclasses import dataclass, asdict
from copy import deepcopy
import json
import os
from pathlib import Path
import numpy as np


from .network import ConditionalUNet


class Flow:
    def __init__(self, dimension, context_dim, config, rng, lower=-1., upper=1.):
        self.dimension = dimension
        self.lower=np.broadcast_to(lower,(dimension,)).copy();self.upper=np.broadcast_to(upper,(dimension,)).copy()
        self.net = ConditionalUNet(dimension,context_dim,config.hidden,config.down_dims,rng,config.weight_decay,config.warmup,config.total_updates,config.lr)

    def forward(self, x, t, context, record=False):
        return self.net.forward(x,t,context,record=record)

    def sample(self, context, rng, steps):
        x = rng.normal(size=(len(context), self.dimension))
        dt = 1 / steps
        for i in range(steps):
            t = np.full((len(x), 1), i*dt)
            v, _ = self.forward(x, t, context)
            v2, _ = self.forward(x+dt*v, t+dt, context)
            x += .5*dt*(v+v2)
        return self.lower+(np.clip(x,-1,1)+1)*.5*(self.upper-self.lower)

    def train(self, context, y, rng, lr):
        y=2*(y-self.lower)/(self.upper-self.lower)-1
        z = rng.normal(size=y.shape)
        t = rng.random((len(y), 1))
        v, cache = self.forward((1-t)*z+t*y, t, context,record=True)
        error = v - (y-z)
        loss = float(np.mean(error**2))
        self.net.backward(2*error/error.size, cache, lr)
        return loss


class Replay:
    """One uniform, accumulating transition buffer. No generated targets."""
    fields = ('observations', 'actions', 'features', 'next_observations', 'terminated', 'truncated')

    def __init__(self):
        self.size = 0
        self.capacity = 0
        self.arrays = {}

    def add(self, observation, action, feature, next_observation, terminated=False, truncated=False):
        values = dict(zip(self.fields, (observation, action, feature, next_observation, terminated, truncated)))
        values = {k: np.asarray(v).copy() for k, v in values.items()}
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError('Nonfinite transition')
        if self.size == self.capacity:
            capacity = max(256, 2*self.capacity)
            for k, value in values.items():
                arr = np.empty((capacity,)+value.shape, dtype=value.dtype)
                if self.size:
                    arr[:self.size] = self.arrays[k][:self.size]
                self.arrays[k] = arr
            self.capacity = capacity
        for k, value in values.items():
            self.arrays[k][self.size] = value
        self.size += 1

    def sample(self, n, rng):
        if not self.size:
            raise ValueError('Empty replay')
        ids = rng.integers(self.size, size=n)
        return {k: v[ids] for k, v in self.arrays.items()}


@dataclass
class Config:
    obs_dim: int
    feature_dim: int
    action_kind: str = 'discrete'
    actions: int = 2
    low: tuple = (-1.,)
    high: tuple = (1.,)
    gamma: float = .99
    hidden: int = 256
    down_dims: tuple = (256,512,1024)
    feature_low: tuple = None
    feature_high: tuple = None
    warmup: int = 500
    total_updates: int = 100000
    weight_decay: float = 1e-6
    elites: int = 1
    batch_size: int = 2048
    ode_steps: int = 24
    lr: float = .0003
    tau: float = .005
    candidates: int = 1000
    gram_samples: int = 128
    ridge: float = .001
    tilt: float = 2.
    epsilon: float = .1
    seed: int = 0

    def __post_init__(self):
        if not 0 <= self.gamma < 1 or not 0 < self.tau <= 1:
            raise ValueError('Invalid discount/target update')
        if self.action_kind not in ('discrete', 'continuous'):
            raise ValueError('Invalid action kind')
        if min(self.obs_dim, self.feature_dim, self.hidden, self.batch_size, self.ode_steps, self.candidates, self.gram_samples) < 1:
            raise ValueError('Dimensions/counts must be positive')
        if self.ridge <= 0 or self.lr <= 0 or not 0 <= self.epsilon <= 1 or self.tilt < 0:
            raise ValueError('Invalid training/acquisition setting')
        if self.action_kind == 'discrete' and self.actions < 2:
            raise ValueError('At least two actions required')
        if self.action_kind == 'continuous':
            low, high = np.asarray(self.low), np.asarray(self.high)
            if low.ndim != 1 or low.shape != high.shape or not low.size or not np.isfinite([low, high]).all() or np.any(low >= high):
                raise ValueError('Invalid action bounds')


def bellman_target(features, successors, terminated, gamma):
    # Unnormalized DiSPO successor features; true terminals have zero future features.
    return features + gamma*(1-np.asarray(terminated, float)[:, None])*successors


class Learner:
    schema = 'dispo-fm-unet-v2'

    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        low=np.zeros(config.feature_dim) if config.feature_low is None else np.asarray(config.feature_low)
        high=np.ones(config.feature_dim) if config.feature_high is None else np.asarray(config.feature_high)
        if low.shape!=(config.feature_dim,) or high.shape!=low.shape or np.any(high<=low):raise ValueError('Invalid feature bounds')
        if config.hidden%2 or min(config.down_dims)<1 or config.elites<1 or config.elites>config.candidates:raise ValueError('Invalid network/planner configuration')
        self.flow = Flow(config.feature_dim, config.obs_dim, config, self.rng,low/(1-config.gamma),high/(1-config.gamma))
        self.target = deepcopy(self.flow)
        context = config.obs_dim + config.feature_dim
        action_dim=config.actions if config.action_kind=='discrete' else len(config.low)
        self.readout = Flow(action_dim,context,config,self.rng)
        self.readout_target=deepcopy(self.readout)
        self.replay = Replay()
        self.round = 0
        self.env_steps = 0
        self.history = []

    def update(self):
        c = self.config
        batch = self.replay.sample(c.batch_size, self.rng)
        # Fresh detached bootstrap sample on every replay use, shared by both losses.
        successors = self.target.sample(batch['next_observations'], self.rng, c.ode_steps)
        y = bellman_target(batch['features'], successors, batch['terminated'], c.gamma)
        flow_loss = self.flow.train(batch['observations'], y, self.rng, c.lr)
        context = np.concatenate([batch['observations'], y], axis=1)
        if c.action_kind == 'discrete':
            actions=2*np.eye(c.actions)[batch['actions'].astype(int)]-1
        else:
            actions=2*(batch['actions']-np.asarray(c.low))/(np.asarray(c.high)-c.low)-1
        readout_loss=self.readout.train(context,actions,self.rng,c.lr)
        for online,target in ((self.flow,self.target),(self.readout,self.readout_target)):
            for k in online.net.p:
                target.net.p[k] *= 1-c.tau
                target.net.p[k] += c.tau*online.net.p[k]
        return dict(flow_loss=flow_loss, readout_loss=readout_loss)

    def outcomes(self, observation, count, rng=None):
        rng = self.rng if rng is None else rng
        return self.target.sample(np.repeat(np.asarray(observation)[None], count, axis=0), rng, self.config.ode_steps)

    def choose(self, observation, method='coverage', reward=None, rng=None):
        """Called anew at every visited state, in both collection and evaluation."""
        c = self.config
        rng = self.rng if rng is None else rng
        candidates = self.outcomes(observation, c.candidates, rng)
        if reward is not None:
            ids=np.argsort(-(candidates @ np.asarray(reward)))[:c.elites]
            return candidates[ids].mean(0)
        if method == 'uniform':
            return candidates[rng.integers(len(candidates))]
        if method != 'coverage':
            raise ValueError('Expected uniform or coverage')
        # Independent reference draw avoids self-normalizing candidate leverage.
        reference = self.outcomes(observation, c.gram_samples, rng)
        gram = c.ridge*np.eye(c.feature_dim) + reference.T @ reference/len(reference)
        leverage = np.einsum('bi,ib->b', candidates, np.linalg.solve(gram, candidates.T))
        logw = c.tilt*np.log1p(np.maximum(leverage, 0))
        weights = np.exp(logw-logw.max())
        return candidates[rng.choice(len(candidates), p=weights/weights.sum())]

    def action(self, observation, outcome, rng=None):
        rng = self.rng if rng is None else rng
        c = self.config
        context = np.concatenate([observation, outcome])[None]
        sample=self.readout_target.sample(context,rng,c.ode_steps)[0]
        if c.action_kind=='discrete':return int(np.argmax(sample))
        return np.asarray(c.low)+(sample+1)*.5*(np.asarray(c.high)-c.low)

    def collect(self, env, feature, steps, reset_context=0, method='coverage', warmup=False, episode_steps=64):
        """Only reset_to and step are used. Reset distribution stays fixed."""
        c = self.config
        def reset():
            return env.reset_to(reset_context, seed=int(self.rng.integers(2**31)))
        obs = reset()
        age = 0
        for _ in range(steps):
            if warmup or self.rng.random() < c.epsilon:
                action = int(self.rng.integers(c.actions)) if c.action_kind == 'discrete' else self.rng.uniform(c.low, c.high)
            else:
                action = self.action(obs, self.choose(obs, method))
            nxt, _, terminal, truncated, _ = env.step(action)
            age += 1
            truncated = bool(truncated or age >= episode_steps)
            self.replay.add(obs, action, feature(obs, action), nxt, bool(terminal), truncated)
            self.env_steps += 1
            obs = reset() if terminal or truncated else nxt
            if terminal or truncated:
                age = 0

    def save(self, path, run_spec=None):
        """Atomic complete checkpoint. No pickle; checkpoints contain raw transitions."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(schema=self.schema, config=asdict(self.config), rng=self.rng.bit_generator.state,
                    round=self.round, env_steps=self.env_steps, history=self.history, run_spec=run_spec)
        nets = {'flow': self.flow.net, 'target': self.target.net,
                'readout': self.readout.net, 'readout_target': self.readout_target.net}
        arrays = {f'replay/{k}': v[:self.replay.size] for k, v in self.replay.arrays.items()}
        for name, net in nets.items():
            meta[name+'_steps'] = net.steps
            for group in ('p', 'm', 'v'):
                arrays.update({f'{name}/{group}/{k}': v for k, v in getattr(net, group).items()})
        arrays['metadata'] = np.array(json.dumps(meta))
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            if meta.get('schema') != cls.schema:
                raise ValueError('Requires a v2 FM U-Net checkpoint')
            obj = cls(Config(**meta['config']))
            obj.rng.bit_generator.state = meta['rng']
            obj.round, obj.env_steps, obj.history = meta['round'], meta['env_steps'], meta['history']
            nets = {'flow': obj.flow.net, 'target': obj.target.net,
                    'readout': obj.readout.net, 'readout_target': obj.readout_target.net}
            for name, net in nets.items():
                net.steps = meta[name+'_steps']
                for group in ('p', 'm', 'v'):
                    for k in net.p:
                        getattr(net, group)[k] = data[f'{name}/{group}/{k}'].copy()
            obj.replay.arrays = {k: data[f'replay/{k}'].copy() for k in Replay.fields if f'replay/{k}' in data}
            obj.replay.size = len(obj.replay.arrays['actions']) if obj.replay.arrays else 0
            obj.replay.capacity = obj.replay.size
        return obj, meta['run_spec']
