"""Transition-based DiSPO updates with conditional flow matching."""
from dataclasses import dataclass, asdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import torch

from .network import ConditionalUNet


def resolve_device(device):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable in this PyTorch installation')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device {device.index} is not visible')
        device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Expected a CPU or CUDA device')
    return device


def elite_mean(candidates, scores, elites):
    """Select independently for each reward, breaking ties by candidate order."""
    ids = (scores.argmax(dim=-1, keepdim=True) if elites == 1
           else scores.argsort(dim=-1, descending=True, stable=True)[..., :elites])
    if candidates.ndim == 2:
        return candidates[ids].mean(dim=-2)
    banks = torch.arange(len(candidates), device=candidates.device)[:, None, None]
    return candidates[banks, ids].mean(dim=-2)


class Flow:
    def __init__(self, dimension, context_dim, config, generator, device, dtype, lower=-1., upper=1., outcome=False):
        self.dimension = dimension
        self.outcome = outcome
        self.config = config
        self.steps = 0
        self.lower = torch.as_tensor(lower, device=device, dtype=dtype).expand(dimension)
        self.upper = torch.as_tensor(upper, device=device, dtype=dtype).expand(dimension)
        self.span = self.upper - self.lower
        self.net = ConditionalUNet(dimension, context_dim, config.hidden, config.down_dims,
                                   generator, dtype).to(device)
        self.ema = deepcopy(self.net).requires_grad_(False).eval()
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=config.lr,
                                           weight_decay=config.weight_decay,
                                           fused=device.type == 'cuda')
        self.solver = None

    @torch.no_grad()
    def sample(self, context, rng, count=None, guidance=None):
        solver = (self.config.ode_solver, self.config.ode_steps, self.outcome and self.config.outcome_simplex,
                  self.outcome and self.config.outcome_step_clip)
        if self.solver != solver:
            self.solver = solver
            self.solver_steps = self.config.ode_steps
            self.sample_graphs = {}
            time_grid = torch.arange(self.solver_steps + 1, device=context.device, dtype=context.dtype)
            self.time_embeddings = self.ema.time_embedding(time_grid / self.solver_steps).split(1)
        x = torch.randn((len(context) if count is None else count, self.dimension),
                        device=context.device, dtype=context.dtype, generator=rng)
        if guidance is not None:
            return self._integrate(x, context, guidance)
        # Collection and bootstrap use fixed shapes. Keep graph memory bounded for other callers.
        key = (len(context), len(x))
        bootstrap = len(context) == len(x) == self.config.batch_size
        collection = len(context) == 1 and len(x) in (1, self.config.candidates + self.config.gram_samples)
        if not x.is_cuda or not (bootstrap or collection):
            return self._integrate(x, context)
        if key not in self.sample_graphs:
            if len(self.sample_graphs) >= 3:
                return self._integrate(x, context)
            self.sample_graphs[key] = self._capture_sample(x, context)
        graph, static_x, static_context, result = self.sample_graphs[key]
        static_x.copy_(x)
        static_context.copy_(context)
        graph.replay()
        return result.clone()

    def _capture_sample(self, x, context):
        # Random noise stays outside capture and EMA parameters are updated in place.
        static_x = x.clone()
        static_context = context.clone()
        with torch.cuda.device(x.device):
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            current = torch.cuda.current_stream()
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                self._integrate(static_x, static_context)
            current.wait_stream(stream)
            with torch.cuda.graph(graph, stream=stream):
                result = self._integrate(static_x, static_context)
        return graph, static_x, static_context, result

    def _integrate(self, x, context, guidance=None):
        # A single context can condition a whole bank without repeated embedding/FiLM work.
        dt = 1 / self.solver_steps
        context_embedding = self.ema.context_embedding(context)
        if guidance is not None:
            # A score shift t*g induces the velocity shift (1-t)*g for
            # x_t = (1-t)*noise + t*outcome, including the limit at t=0.
            for step, embedding in enumerate(self.time_embeddings[:-1]):
                remaining = (self.solver_steps - step) * dt
                v = self.ema.velocity(x, embedding, context_embedding) + remaining * guidance
                predicted = x + dt * v
                if self.solver[0] == 'heun':
                    next_v = self.ema.velocity(predicted, self.time_embeddings[step + 1], context_embedding)
                    next_v = next_v + (remaining - dt) * guidance
                    predicted = x + .5 * dt * (v + next_v)
                x = predicted
            return self.lower + (x.clamp(-1, 1) + 1) * .5 * self.span
        if self.solver[3]:
            def clipped_velocity(value, embedding, remaining):
                velocity = self.ema.velocity(value, embedding, context_embedding)
                # For straight-line FM, x + (1 - t) * v predicts the clean endpoint.
                endpoint = (value + remaining * velocity).clamp(-1, 1)
                return (endpoint - value) / remaining

            for step, t in enumerate(self.time_embeddings[:-1]):
                remaining = (self.solver_steps - step) * dt
                v = clipped_velocity(x, t, remaining)
                predicted = x + dt * v
                # The endpoint velocity is undefined at t=1 after clipping.
                if self.solver[0] == 'heun' and step + 1 < self.solver_steps:
                    next_v = clipped_velocity(predicted, self.time_embeddings[step + 1], remaining - dt)
                    x = x + .5 * dt * (v + next_v)
                else:
                    x = predicted
        elif self.solver[0] == 'euler':
            for t in self.time_embeddings[:-1]:
                x = x + dt * self.ema.velocity(x, t, context_embedding)
        else:
            for t, next_t in zip(self.time_embeddings, self.time_embeddings[1:]):
                v = self.ema.velocity(x, t, context_embedding)
                v_next = self.ema.velocity(x + dt * v, next_t, context_embedding)
                x = x + (.5 * dt) * (v + v_next)
        if self.solver[2]:
            # Euclidean projection of normalized occupancy onto the unit simplex.
            occupancy = (x + 1) * .5
            ordered = occupancy.sort(dim=1, descending=True).values
            ranks = torch.arange(1, self.dimension + 1, device=x.device, dtype=x.dtype)
            thresholds = (ordered.cumsum(dim=1) - 1) / ranks
            active = (ordered > thresholds).sum(dim=1, keepdim=True).clamp_min(1)
            threshold = thresholds.gather(1, active - 1)
            occupancy = (occupancy - threshold).clamp_min(0)
            return occupancy / (1 - self.config.gamma)
        return self.lower + (x.clamp(-1, 1) + 1) * .5 * self.span

    def loss(self, context, y, rng, weights=None):
        y = 2 * (y - self.lower) / self.span - 1
        z = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=rng)
        t = torch.rand((len(y), 1), device=y.device, dtype=y.dtype, generator=rng)
        velocity = self.net(torch.lerp(z, y, t), t, context)
        if weights is None:
            return torch.nn.functional.mse_loss(velocity, y - z)
        return (weights * (velocity - (y - z)).square().mean(dim=1)).sum()

    def backward(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1., error_if_nonfinite=True)

    def step(self):
        c = self.config
        if self.steps < c.warmup:
            lr = c.lr * self.steps / max(1, c.warmup)
        else:
            progress = min(1., (self.steps - c.warmup) / max(1, c.total_updates - c.warmup))
            lr = c.lr * .5 * (1 + math.cos(math.pi * progress))
        self.optimizer.param_groups[0]['lr'] = lr
        self.optimizer.step()
        self.steps += 1
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def update_ema(self):
        torch._foreach_lerp_(list(self.ema.parameters()), list(self.net.parameters()), self.config.tau)


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
    ode_steps: int = 10
    ode_solver: str = 'euler'
    outcome_simplex: bool = False
    outcome_step_clip: bool = False
    lr: float = .0003
    tau: float = .005
    candidates: int = 1000
    gram_samples: int = 128
    ridge: float = .001
    tilt: float = 2.
    uniform_mix: float = .1
    goal_pool_multiplier: int = 1
    coverage_state_rbf: bool = False
    coverage_weighting: str = 'linear'
    gram_ema_rate: float = .995
    epsilon: float = .1
    seed: int = 0

    def __post_init__(self):
        if not 0 <= self.gram_ema_rate < 1:
            raise ValueError('gram_ema_rate must be in [0,1)')
        if self.goal_pool_multiplier < 1:
            raise ValueError('goal_pool_multiplier must be positive')
        if self.coverage_weighting not in ('linear', 'power'):
            raise ValueError('Invalid coverage weighting')
        if not 0 <= self.uniform_mix <= 1:
            raise ValueError('uniform_mix must be between 0 and 1')
        if not 0 <= self.gamma < 1 or not 0 < self.tau <= 1:
            raise ValueError('Invalid discount/target update')
        if self.action_kind not in ('discrete', 'continuous'):
            raise ValueError('Invalid action kind')
        if self.ode_solver not in ('heun', 'euler'):
            raise ValueError('Expected heun or euler solver')
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
    if torch.is_tensor(features):
        terminated = torch.as_tensor(terminated, dtype=features.dtype, device=features.device)
    else:
        terminated = np.asarray(terminated, dtype=float)
    return features + gamma * (1 - terminated[:, None]) * successors


class Learner:
    schema = 'dispo-fm-unet-v3'

    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = resolve_device(device)
        self.dtype = torch.float32 if self.device.type == 'cuda' else torch.float64
        self.rng = np.random.default_rng(config.seed)
        self.model_rng = torch.Generator(device=self.device).manual_seed(config.seed)
        init_rng = torch.Generator().manual_seed(config.seed)
        low = np.zeros(config.feature_dim) if config.feature_low is None else np.asarray(config.feature_low)
        high = np.ones(config.feature_dim) if config.feature_high is None else np.asarray(config.feature_high)
        if low.shape != (config.feature_dim,) or high.shape != low.shape or np.any(high <= low):
            raise ValueError('Invalid feature bounds')
        if config.outcome_simplex and (config.action_kind != 'discrete' or np.any(low != 0) or np.any(high != 1)):
            raise ValueError('Outcome simplex requires nonterminating one-hot features with bounds [0,1]')
        if config.hidden % 2 or not config.down_dims or min(config.down_dims) < 1 or not 1 <= config.elites <= config.candidates:
            raise ValueError('Invalid network/planner configuration')
        self.flow = Flow(config.feature_dim, config.obs_dim, config, init_rng, self.device, self.dtype,
                         low / (1 - config.gamma), high / (1 - config.gamma), outcome=True)
        action_dim = config.actions if config.action_kind == 'discrete' else len(config.low)
        self.readout = Flow(action_dim, config.obs_dim + config.feature_dim, config,
                            init_rng, self.device, self.dtype)
        self.action_low = self.tensor(config.low)
        self.action_span = self.tensor(config.high) - self.action_low
        if config.action_kind == 'discrete':
            self.action_targets = 2 * torch.eye(config.actions, device=self.device, dtype=self.dtype) - 1
        self.ridge = config.ridge * torch.eye(config.feature_dim, device=self.device, dtype=torch.float64)
        self.replay = Replay()
        self.gram_ema = None
        self.round = 0
        self.env_steps = 0
        self.history = []

    def tensor(self, value):
        dtype = np.float32 if self.dtype == torch.float32 else np.float64
        return torch.as_tensor(np.asarray(value, dtype=dtype)).to(self.device, non_blocking=True)

    @torch.no_grad()
    def _training_weights(self, observations, targets):
        targets = targets.double()
        joint_block = torch.cat((torch.ones_like(targets[:, :1]), targets), dim=1)
        if self.config.coverage_state_rbf:
            if self.config.obs_dim != 104:
                raise ValueError('State RBF coverage requires Four Rooms observations')
            from environments import fourrooms_cells
            coordinates = torch.as_tensor(fourrooms_cells(), device=targets.device, dtype=targets.dtype)
            centers = coordinates.new_tensor([[0, 0], [0, 10], [10, 0], [10, 10]])
            state = coordinates.index_select(0, observations.argmax(dim=1))
            state_features = torch.exp(-(state[:, None] - centers).square().sum(dim=2) / (2 * 5.0 ** 2))
            joint = (state_features[:, :, None] * joint_block[:, None, :]).flatten(1)
            batch_gram = joint.T @ joint / len(targets)
        else:
            # One-hot states make the joint Gram block diagonal.
            states = observations.argmax(dim=1)
            membership = observations.T.double()
            width = joint_block.shape[1]
            outer = joint_block[:, :, None] * joint_block[:, None, :]
            batch_gram = (membership @ outer.flatten(1)).reshape(-1, width, width) / len(targets)
        if self.gram_ema is not None and self.gram_ema.shape != batch_gram.shape:
            raise ValueError('Coverage Gram shape does not match state features')
        if self.gram_ema is None:
            self.gram_ema = batch_gram.clone()
        else:
            rate = self.config.gram_ema_rate
            self.gram_ema.mul_(rate).add_(batch_gram, alpha=1 - rate)
        width = self.gram_ema.shape[-1]
        ridge = self.config.ridge * torch.eye(width, device=targets.device, dtype=targets.dtype)
        inverse = torch.linalg.inv(self.gram_ema + ridge)
        if self.config.coverage_state_rbf:
            leverage = ((joint @ inverse) * joint).sum(dim=1).clamp_min(0)
        else:
            projected = (inverse.index_select(0, states) @ joint_block[:, :, None]).squeeze(-1)
            leverage = (joint_block * projected).sum(dim=1).clamp_min(0)
        if self.config.coverage_weighting == 'linear':
            total = leverage.sum()
            weights = torch.where(total > 0, leverage / total.clamp_min(torch.finfo(leverage.dtype).tiny),
                                  torch.full_like(leverage, 1 / len(leverage)))
        else:
            weights = (self.config.tilt * leverage.log1p()).softmax(dim=0)
        mix = self.config.uniform_mix
        return ((1 - mix) * weights + mix / len(weights)).to(self.dtype)

    def update_tensor(self, coverage=False, indices=None):
        c = self.config
        pool_size = c.batch_size * (c.goal_pool_multiplier if coverage else 1)
        if indices is None:
            samples = self.replay.sample(pool_size, self.rng)
        else:
            if len(indices) != pool_size:
                raise ValueError('Batch indices must match the training pool size')
            samples = {key: values[indices] for key, values in self.replay.arrays.items()}
        batch = {key: self.tensor(value) for key, value in samples.items()
                 if key != 'truncated'}
        # Draw the detached bootstrap once and share the physical target between models.
        successors = self.flow.sample(batch['next_observations'], self.model_rng)
        y = bellman_target(batch['features'], successors, batch['terminated'], c.gamma)
        weight_args = {'weights': self._training_weights(batch['observations'], y)} if coverage else {}
        # Preserve the outcome Bellman distribution; tilt only readout training.
        flow_loss = self.flow.loss(batch['observations'][:c.batch_size], y[:c.batch_size], self.model_rng)
        self.flow.backward(flow_loss)
        context = torch.cat((batch['observations'], y), dim=1)
        if c.action_kind == 'discrete':
            actions = self.action_targets.index_select(0, batch['actions'].long())
        else:
            actions = 2 * (batch['actions'] - self.action_low) / self.action_span - 1
        if pool_size > c.batch_size:
            indices = torch.multinomial(weight_args['weights'], c.batch_size, replacement=False, generator=self.model_rng)
            context, actions = context[indices], actions[indices]
            weight_args = {}
        readout_loss = self.readout.loss(context, actions, self.model_rng, **weight_args)
        self.readout.backward(readout_loss)
        self.flow.step()
        self.readout.step()
        self.flow.update_ema()
        self.readout.update_ema()
        return torch.stack((flow_loss.detach(), readout_loss.detach()))

    def update(self):
        return dict(zip(('flow_loss', 'readout_loss'), self.update_tensor().tolist()))

    def _sampling_rng(self, rng):
        if rng is None:
            return self.model_rng
        return torch.Generator(device=self.device).manual_seed(int(rng.integers(2**63)))

    def outcomes(self, observation, count, rng=None):
        observation = self.tensor(observation)[None]
        return self.flow.sample(observation, self._sampling_rng(rng), count).cpu().numpy()

    def _choose(self, observation, method, reward, rng):
        c = self.config
        if reward is not None:
            candidates = self.flow.sample(observation, rng, c.candidates)
            return elite_mean(candidates, candidates @ self.tensor(reward), c.elites)
        if method == 'uniform':
            # A uniform pick from an i.i.d. bank has the distribution of one draw.
            return self.flow.sample(observation, rng)[0]
        if method != 'coverage':
            raise ValueError('Expected uniform or coverage')
        samples = self.flow.sample(observation, rng, c.candidates + c.gram_samples)
        candidates, reference = samples.double().split((c.candidates, c.gram_samples))
        gram = self.ridge + reference.T @ reference / len(reference)
        projected, _ = torch.linalg.solve_ex(gram, candidates.T)
        leverage = (candidates * projected.T).sum(dim=1).clamp_min(0)
        weights = (c.tilt * leverage.log1p()).softmax(dim=0)
        index = torch.multinomial(weights, 1, generator=rng)
        return candidates.index_select(0, index)[0].to(self.dtype)

    def choose(self, observation, method='coverage', reward=None, rng=None):
        observation = self.tensor(observation)[None]
        return self._choose(observation, method, reward, self._sampling_rng(rng)).cpu().numpy()

    def _action(self, observation, outcome, rng):
        context = torch.cat((observation, outcome[None]), dim=1)
        sample = self.readout.sample(context, rng)[0]
        if self.config.action_kind == 'discrete':
            return int(sample.argmax())
        return (self.action_low + (sample + 1) * .5 * self.action_span).cpu().numpy()

    def action(self, observation, outcome, rng=None):
        return self._action(self.tensor(observation)[None], self.tensor(outcome), self._sampling_rng(rng))

    def collect(self, env, feature, steps, reset_context=0, method='uniform', warmup=False, episode_steps=64):
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
                observation = self.tensor(obs)[None]
                outcome = self._choose(observation, method, None, self.model_rng)
                action = self._action(observation, outcome, self.model_rng)
            nxt, _, terminal, truncated, _ = env.step(action)
            age += 1
            truncated = bool(truncated or age >= episode_steps)
            self.replay.add(obs, action, feature(obs, action), nxt, bool(terminal), truncated)
            self.env_steps += 1
            obs = reset() if terminal or truncated else nxt
            if terminal or truncated:
                age = 0

    def _checkpoint_flows(self):
        return {'flow': (self.flow, self.flow.net), 'target': (None, self.flow.ema),
                'readout': (self.readout, self.readout.net), 'readout_target': (None, self.readout.ema)}

    def save(self, path, run_spec=None):
        """Atomically replace the checkpoint with replay, parameters, and training state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(schema=self.schema, config=asdict(self.config), rng=self.rng.bit_generator.state,
                    rng_device=self.device.type, round=self.round, env_steps=self.env_steps,
                    history=self.history, run_spec=run_spec)
        arrays = {f'replay/{k}': v[:self.replay.size] for k, v in self.replay.arrays.items()}
        arrays['model_rng'] = self.model_rng.get_state().numpy()
        if self.gram_ema is not None:
            arrays['gram_ema'] = self.gram_ema.detach().cpu().numpy()
        for name, (flow, net) in self._checkpoint_flows().items():
            meta[name + '_steps'] = 0 if flow is None else flow.steps
            for key, parameter, transpose in net.checkpoint_parameters():
                arrays[f'{name}/p/{key}'] = self._checkpoint_array(parameter, transpose)
                if flow is not None and flow.steps:
                    state = flow.optimizer.state[parameter]
                    arrays[f'{name}/m/{key}'] = self._checkpoint_array(state['exp_avg'], transpose)
                    arrays[f'{name}/v/{key}'] = self._checkpoint_array(state['exp_avg_sq'], transpose)
        arrays['metadata'] = np.array(json.dumps(meta))
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @staticmethod
    def _checkpoint_array(tensor, transpose):
        tensor = tensor.detach().cpu()
        return (tensor.T if transpose else tensor).numpy()

    @classmethod
    @torch.no_grad()
    def load(cls, path, device='cpu'):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            if meta.get('schema') not in (cls.schema, 'dispo-fm-unet-v2'):
                raise ValueError('Requires a v2 or v3 FM U-Net checkpoint')
            config = meta['config']
            config.setdefault('uniform_mix', 0.)
            config.setdefault('coverage_weighting', 'power')
            spec = meta.get('run_spec') or {}
            config.setdefault('outcome_simplex', spec.get('environment') == 'garnet')
            if spec.get('environment') == 'fourrooms':
                config['outcome_simplex'] = False
                config['outcome_step_clip'] = True
            if 'ode_solver' not in config:
                # Migrate the old default while retaining custom legacy Heun step counts.
                if config.get('ode_steps', 24) == 24:
                    config.pop('ode_steps', None)
                else:
                    config['ode_solver'] = 'heun'
            obj = cls(Config(**config), device)
            if 'gram_ema' in data:
                obj.gram_ema = torch.as_tensor(data['gram_ema'], dtype=torch.float64, device=obj.device)
            obj.rng.bit_generator.state = meta['rng']
            if 'model_rng' in data:
                rng = data['model_rng']
                if meta['rng_device'] == obj.device.type:
                    obj.model_rng.set_state(torch.from_numpy(rng))
                else:
                    seed = int.from_bytes(hashlib.blake2b(rng, digest_size=8).digest(), 'little')
                    obj.model_rng.manual_seed(seed)
            obj.round, obj.env_steps, obj.history = meta['round'], meta['env_steps'], meta['history']
            for name, (flow, net) in obj._checkpoint_flows().items():
                if flow is not None:
                    flow.steps = meta[name + '_steps']
                for key, parameter, transpose in net.checkpoint_parameters():
                    weight = torch.from_numpy(data[f'{name}/p/{key}'])
                    parameter.copy_(weight.T if transpose else weight)
                    if flow is not None and flow.steps:
                        state = flow.optimizer.state[parameter]
                        state['step'] = torch.tensor(float(flow.steps), device=obj.device, dtype=torch.float32)
                        for group, field in (('m', 'exp_avg'), ('v', 'exp_avg_sq')):
                            moment = torch.from_numpy(data[f'{name}/{group}/{key}'])
                            state[field] = torch.empty_like(parameter)
                            state[field].copy_(moment.T if transpose else moment)
            obj.replay.arrays = {k: data[f'replay/{k}'] for k in Replay.fields if f'replay/{k}' in data}
            obj.replay.size = len(obj.replay.arrays['actions']) if obj.replay.arrays else 0
            obj.replay.capacity = obj.replay.size
        return obj, meta['run_spec']
