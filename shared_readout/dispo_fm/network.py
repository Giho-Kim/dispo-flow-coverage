"""NumPy reverse-mode implementation of DiSPO's conditional linear U-Net."""
import numpy as np

class Tape:
    def __init__(self,record=True): self.ops=[];self.leaves=[];self.record=record
    def value(self,x):
        node=Node(np.asarray(x),self)
        if self.record:self.leaves.append(node)
        return node
    def backward(self,out,gradient):
        out.grad=gradient
        for node,fn in reversed(self.ops):
            if node.grad is not None: fn(node.grad)

class Node:
    def __init__(self,x,tape): self.x=x;self.tape=tape;self.grad=None
    def addgrad(self,g): self.grad=g if self.grad is None else self.grad+g

def op(values,forward,backward):
    tape=values[0].tape;out=Node(forward(*[v.x for v in values]),tape)
    def back(g):
        for v,d in zip(values,backward(g,*[v.x for v in values])):v.addgrad(d)
    if tape.record:tape.ops.append((out,back))
    return out

def add(a,b):return op([a,b],lambda x,y:x+y,lambda g,x,y:(g,g))
def mul(a,b):return op([a,b],lambda x,y:x*y,lambda g,x,y:(g*y,g*x))
def cat(a,b):return op([a,b],lambda x,y:np.concatenate([x,y],1),lambda g,x,y:(g[:,:x.shape[1]],g[:,x.shape[1]:]))

class ConditionalUNet:
    def __init__(self,dimension,context_dim,embed,downs,rng,weight_decay=1e-6,warmup=500,total=100000,lr=3e-4):
        self.p={};self.m={};self.v={};self.steps=0
        self.dimension=dimension;self.context_dim=context_dim;self.embed=embed;self.downs=tuple(downs)
        self.weight_decay=weight_decay;self.warmup=warmup;self.total=total;self.lr=lr;self.rng=rng
        # Allocate parameters by running the exact topology once.
        self.forward(np.zeros((1,dimension)),np.zeros((1,1)),np.zeros((1,context_dim)))
        self.m={k:np.zeros_like(v) for k,v in self.p.items()};self.v={k:np.zeros_like(v) for k,v in self.p.items()}

    def param(self,key,shape,kind='weight'):
        if key not in self.p:
            self.p[key]=(np.ones(shape) if kind=='scale' else np.zeros(shape) if kind=='bias' else self.rng.normal(size=shape)/np.sqrt(shape[0]))
        return self.tape.value(self.p[key])

    def dense(self,x,width,name):
        w=self.param(name+'/w',(x.x.shape[1],width));b=self.param(name+'/b',(width,), 'bias')
        return op([x,w,b],lambda x,w,b:x@w+b,lambda g,x,w,b:(g@w.T,x.T@g,g.sum(0)))

    def block(self,x,width,name):
        x=self.dense(x,width,name+'/dense')
        def forward(a):
            centered=a-a.mean(1,keepdims=True);return centered/np.sqrt((centered**2).mean(1,keepdims=True)+1e-6)
        def backward(g,a):
            inv=1/np.sqrt(a.var(1,keepdims=True)+1e-6);z=(a-a.mean(1,keepdims=True))*inv
            return (inv*(g-g.mean(1,keepdims=True)-z*(g*z).mean(1,keepdims=True)),)
        x=op([x],forward,backward)
        scale=self.param(name+'/ln/scale',(width,), 'scale');bias=self.param(name+'/ln/bias',(width,), 'bias')
        x=op([x,scale,bias],lambda a,s,b:a*s+b,lambda g,a,s,b:(g*s,(g*a).sum(0),g.sum(0)))
        def sigmoid(a):return 1/(1+np.exp(-np.clip(a,-60,60)))
        return op([x],lambda a:a*sigmoid(a),lambda g,a:(g*(sigmoid(a)+a*sigmoid(a)*(1-sigmoid(a))),))

    def residual(self,x,cond,width,name):
        y=self.block(x,width,name+'/first');mod=self.dense(cond,2*width,name+'/film')
        gamma=op([mod],lambda a:a[:,:width],lambda g,a:(np.concatenate([g,np.zeros_like(g)],1),))
        beta=op([mod],lambda a:a[:,width:],lambda g,a:(np.concatenate([np.zeros_like(g),g],1),))
        y=add(mul(gamma,y),beta);y=self.block(y,width,name+'/second')
        return add(y,self.dense(x,width,name+'/skip'))

    def forward(self,x,t,context,record=False):
        self.tape=Tape(record);x=self.tape.value(x);ctx=self.tape.value(context)
        half=self.embed//2;freq=np.exp(np.arange(half)*(-np.log(10000)/max(half-1,1)))
        angles=np.asarray(t).reshape(-1,1)*999*freq[None]
        te=self.tape.value(np.concatenate([np.sin(angles),np.cos(angles)],1))
        width=2*max(self.embed,self.context_dim)
        ctx=self.block(ctx,width,'condition/0');ctx=self.block(ctx,width,'condition/1');ctx=self.dense(ctx,self.embed,'condition/2')
        cond=cat(te,ctx);skips=[]
        for i,width in enumerate(self.downs):
            for j in range(2):x=self.residual(x,cond,width,f'down/{i}/{j}')
            skips.append(x)
        for j in range(2):x=self.residual(x,cond,self.downs[-1],f'mid/{j}')
        for i,width in enumerate(reversed(self.downs[:-1])):
            x=cat(x,skips.pop())
            for j in range(2):x=self.residual(x,cond,width,f'up/{i}/{j}')
        x=self.block(x,self.downs[0],'out/0');x=self.dense(x,self.dimension,'out/1')
        return x.x,(self.tape,x)

    def backward(self,delta,cache,rate=None):
        tape,out=cache;tape.backward(out,delta)
        # All parameter nodes point at the corresponding array; collect repeated uses.
        grads={k:np.zeros_like(v) for k,v in self.p.items()};byid={id(v):k for k,v in self.p.items()}
        for node in tape.leaves:
            key=byid.get(id(node.x))
            if key is not None and node.grad is not None:grads[key]+=node.grad
        norm=np.sqrt(sum((g*g).sum() for g in grads.values()))
        if not np.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        step=self.steps
        lr=self.lr*min(1.,step/max(1,self.warmup)) if step<self.warmup else self.lr*.5*(1+np.cos(np.pi*min(1.,(step-self.warmup)/max(1,self.total-self.warmup))))
        self.steps+=1
        for k,g in grads.items():
            g=g*min(1.,1/max(norm,1e-12))
            self.m[k]=.9*self.m[k]+.1*g;self.v[k]=.999*self.v[k]+.001*g*g
            self.p[k]-=lr*(self.m[k]/(1-.9**self.steps)/(np.sqrt(self.v[k]/(1-.999**self.steps))+1e-8)+self.weight_decay*self.p[k])
