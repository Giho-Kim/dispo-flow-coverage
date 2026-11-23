"""Reference adapters. Only simulator internals access transition dynamics."""
import numpy as np
from actor import ActionSpace

class Garnet:
    def __init__(self,p):self._p=p;self.states=len(p)
    def reset_to(self,context,seed=0):
        self._rng=np.random.default_rng(seed);self._state=int(context)
        return np.eye(self.states)[self._state]
    def sample_batch(self,states,actions,rng):
        return (rng.random((len(states),1))>np.cumsum(self._p[states,actions],axis=1)).sum(1)
    def step(self,action):
        self._state=int(self._rng.choice(self.states,p=self._p[self._state,int(action)]))
        return np.eye(self.states)[self._state],0.,False,False,{}

class ContinuousSystem:
    """Bounded continuous state/action smoke-test environment."""
    def reset_to(self,context,seed=0):
        self._rng=np.random.default_rng(seed);self._x=float(context)
        return np.array([self._x,1.])
    def step(self,action):
        action=np.asarray(action)
        if action.shape!=(1,) or abs(action[0])>1:raise ValueError('Invalid continuous action')
        self._x=float(np.clip(.8*self._x+.2*action[0]+self._rng.normal(scale=.01),-1,1))
        return np.array([self._x,1.]),0.,False,False,{}


def make_experiment(name,seed=0,teacher_count=16,branching=2):
    rng=np.random.default_rng(seed)
    if name=='garnet':
        s,a,b=6,2,branching
        if b not in (1,2): raise ValueError("branching must be 1 or 2")
        while True:
            p=np.zeros((s,a,s))
            for x in range(s):
                for u in range(a):
                    ids=rng.choice(s,b,replace=False);p[x,u,ids]=rng.dirichlet(np.ones(b))
            reach=(p.sum(1)>0)|np.eye(s,dtype=bool)
            for j in range(s):reach|=reach[:,j,None]&reach[None,j,:]
            if reach.all():break
        teachers=[]
        for _ in range(teacher_count):
            table=rng.integers(a,size=s)
            teachers.append(lambda obs,noise,table=table:int(table[int(np.argmax(obs))]))
        return dict(env_factory=lambda:Garnet(p),feature=lambda obs,action:np.eye(s*a)[int(np.argmax(obs))*a+int(action)],
                    feature_dim=s*a,space=ActionSpace('discrete',n=a),contexts=list(range(s)),
                    anchors=np.eye(s),weights=np.full(s,1/s),teachers=teachers)
    if name=='continuous':
        teachers=[]
        for _ in range(teacher_count):
            gain,bias=rng.normal(size=2)
            teachers.append(lambda obs,noise,gain=gain,bias=bias:np.array([np.tanh(gain*obs[0]+bias)]))
        contexts=[-.8,0.,.8]
        return dict(env_factory=ContinuousSystem,
                    feature=lambda obs,action:np.array([obs[0],action[0],obs[0]**2,action[0]**2]),
                    feature_dim=4,space=ActionSpace('continuous',low=[-1.],high=[1.]),contexts=contexts,
                    anchors=np.array([[x,1.] for x in contexts]),weights=np.full(3,1/3),teachers=teachers)
    raise ValueError('Unknown environment')
