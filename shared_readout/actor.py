"""One conditional readout, with a continuous vector goal and either action type."""
from dataclasses import dataclass
import json
import numpy as np

@dataclass
class ActionSpace:
    kind: str
    n: int = 0
    low: object = None
    high: object = None
    def __post_init__(self):
        if self.kind == 'discrete':
            if not isinstance(self.n,int) or self.n < 2: raise ValueError('n must be an integer >= 2')
        elif self.kind == 'continuous':
            self.low=np.asarray(self.low,float);self.high=np.asarray(self.high,float)
            if self.low.ndim!=1 or self.low.shape!=self.high.shape or not self.low.size or not np.isfinite([self.low,self.high]).all() or np.any(self.low>=self.high):
                raise ValueError('Finite vector bounds with low < high required')
        else: raise ValueError('Unknown action space')
    @property
    def size(self): return self.n if self.kind=='discrete' else len(self.low)
    def metadata(self):
        return dict(kind=self.kind,n=self.n,low=None if self.low is None else self.low.tolist(),high=None if self.high is None else self.high.tolist())

class SharedReadout:
    """pi(a | observation, anchor_observation, continuous_goal)."""
    def __init__(self, observation_dim, goal_dim, space, hidden=64, seed=0, continuous_std=.35):
        if min(observation_dim,goal_dim,hidden)<1 or continuous_std<=0: raise ValueError('Invalid dimensions/std')
        self.observation_dim,self.goal_dim,self.hidden=observation_dim,goal_dim,hidden
        self.space=space;self.continuous_std=continuous_std
        self.goal_mean=np.zeros(goal_dim);self.goal_scale=np.ones(goal_dim)
        rng=np.random.default_rng(seed);sizes=[2*observation_dim+goal_dim,hidden,hidden,space.size]
        self.params={}
        for i,(d,e) in enumerate(zip(sizes[:-1],sizes[1:])):
            self.params[f'w{i}']=rng.normal(size=(d,e))*np.sqrt(2/(d+e))
            self.params[f'b{i}']=np.zeros(e)
        self.first={k:np.zeros_like(v) for k,v in self.params.items()}
        self.second={k:np.zeros_like(v) for k,v in self.params.items()};self.updates=0

    def forward(self, observations, anchors, goals, cache=False):
        o=np.atleast_2d(observations);a=np.atleast_2d(anchors);g=np.atleast_2d(goals)
        if o.shape[1]!=self.observation_dim or a.shape!=o.shape or g.shape!=(len(o),self.goal_dim):
            raise ValueError('Expected batched observations, anchors and continuous goal vectors')
        x=np.concatenate([o,a,(g-self.goal_mean)/self.goal_scale],axis=1)
        h1=np.tanh(x@self.params['w0']+self.params['b0'])
        h2=np.tanh(h1@self.params['w1']+self.params['b1'])
        output=h2@self.params['w2']+self.params['b2']
        return (output,(x,h1,h2)) if cache else output

    def act(self, observation, anchor, goal, rng):
        out=self.forward(observation,anchor,goal)[0]
        if self.space.kind=='discrete':
            prob=np.exp(out-out.max());prob/=prob.sum()
            return int(rng.choice(self.space.n,p=prob))
        latent=out+self.continuous_std*rng.normal(size=out.shape)
        return self.space.low+(np.tanh(latent)+1)*.5*(self.space.high-self.space.low)

    def loss_and_grad(self, observations, anchors, goals, actions, weights=None):
        out,(x,h1,h2)=self.forward(observations,anchors,goals,True)
        n=len(out);weights=np.ones(n)/n if weights is None else np.asarray(weights,float)
        if weights.shape!=(n,) or not np.isfinite(weights).all():raise ValueError('Invalid weights')
        if self.space.kind=='discrete':
            actions=np.asarray(actions)
            if actions.shape!=(n,) or not np.issubdtype(actions.dtype,np.integer) or np.any(actions<0) or np.any(actions>=self.space.n):raise ValueError('Invalid actions')
            shifted=out-out.max(1,keepdims=True);logp=shifted-np.log(np.exp(shifted).sum(1,keepdims=True))
            nll=-logp[np.arange(n),actions];delta=np.exp(logp);delta[np.arange(n),actions]-=1
        else:
            actions=np.asarray(actions,float)
            if actions.shape!=out.shape or not np.isfinite(actions).all() or np.any(actions<self.space.low) or np.any(actions>self.space.high):raise ValueError('Invalid bounded actions')
            half=(self.space.high-self.space.low)/2
            y=np.clip((actions-self.space.low)/half-1,-1+1e-7,1-1e-7)
            latent=np.arctanh(y);sigma=self.continuous_std
            nll=(.5*((latent-out)/sigma)**2+np.log(sigma)+.5*np.log(2*np.pi)+np.log(half*(1-y*y))).sum(1)
            delta=(out-latent)/(sigma*sigma)
        loss=float(weights@nll);delta*=weights[:,None]
        grad=dict(w2=h2.T@delta,b2=delta.sum(0))
        delta=(delta@self.params['w2'].T)*(1-h2*h2)
        grad.update(w1=h1.T@delta,b1=delta.sum(0))
        delta=(delta@self.params['w1'].T)*(1-h1*h1)
        grad.update(w0=x.T@delta,b0=delta.sum(0))
        return loss,grad

    def update(self,grad,rate=1e-3,clip=10.):
        norm=np.sqrt(sum(np.sum(g*g) for g in grad.values()));factor=min(1.,clip/max(norm,1e-12))
        if not np.isfinite(norm):raise RuntimeError('Nonfinite gradient')
        self.updates+=1
        for key in self.params:
            g=grad[key]*factor
            self.first[key]=.9*self.first[key]+.1*g;self.second[key]=.999*self.second[key]+.001*g*g
            self.params[key]-=rate*(self.first[key]/(1-.9**self.updates))/(np.sqrt(self.second[key]/(1-.999**self.updates))+1e-8)

    def save(self,path):
        meta=dict(observation_dim=self.observation_dim,goal_dim=self.goal_dim,hidden=self.hidden,
                  continuous_std=self.continuous_std,space=self.space.metadata(),updates=self.updates)
        arrays=dict(self.params,goal_mean=self.goal_mean,goal_scale=self.goal_scale,metadata=np.array(json.dumps(meta)))
        arrays.update({f'first_{k}':v for k,v in self.first.items()});arrays.update({f'second_{k}':v for k,v in self.second.items()})
        np.savez_compressed(path,**arrays)
    @classmethod
    def load(cls,path):
        with np.load(path) as data:
            meta=json.loads(str(data['metadata']));space=ActionSpace(**meta.pop('space'));updates=meta.pop('updates')
            obj=cls(space=space,**meta);obj.updates=updates
            obj.goal_mean=data['goal_mean'];obj.goal_scale=data['goal_scale']
            for key in obj.params:
                obj.params[key]=data[key];obj.first[key]=data[f'first_{key}'];obj.second[key]=data[f'second_{key}']
        return obj
