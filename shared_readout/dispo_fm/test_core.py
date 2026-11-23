import tempfile
import unittest
from pathlib import Path
import numpy as np
from .core import Config,Learner,bellman_target
from .network import ConditionalUNet


def small(kind='discrete'):
    return Learner(Config(2,4,action_kind=kind,hidden=4,down_dims=(4,6,8),warmup=0,
                         batch_size=4,ode_steps=2,candidates=4,gram_samples=4,gamma=.9))

def fill(a):
    for i in range(12):
        action=i%2 if a.config.action_kind=='discrete' else np.array([(-1)**i*.5])
        a.replay.add(np.eye(2)[i%2],action,np.eye(4)[i%4],np.eye(2)[1-i%2])

class Tests(unittest.TestCase):
    def test_gradients_network(self):
        net=ConditionalUNet(2,3,4,(4,6,8),np.random.default_rng(3),warmup=0)
        rng=np.random.default_rng(5);x=rng.normal(size=(2,2));t=rng.random((2,1));c=rng.normal(size=(2,3));direction=rng.normal(size=(2,2))
        y,cache=net.forward(x,t,c,record=True);tape,node=cache;tape.backward(node,direction)
        analytic={}
        for leaf in tape.leaves:
            for k,p in net.p.items():
                if leaf.x is p and leaf.grad is not None:analytic[k]=leaf.grad.copy()
        for key in ('condition/0/dense/w','down/0/0/film/w','mid/1/first/ln/scale','up/0/0/skip/w','out/1/b'):
            p=net.p[key];idx=tuple(0 for _ in p.shape);orig=p[idx];eps=1e-5
            p[idx]=orig+eps;plus=(net.forward(x,t,c)[0]*direction).sum()
            p[idx]=orig-eps;minus=(net.forward(x,t,c)[0]*direction).sum();p[idx]=orig
            self.assertAlmostEqual(analytic[key][idx],(plus-minus)/(2*eps),places=5,msg=key)

    def test_scaling_and_target(self):
        a=small();y=a.outcomes([1.,0.],16)
        self.assertTrue((y>=0).all() and (y<=10+1e-10).all())
        np.testing.assert_allclose(bellman_target(np.ones((2,4)),np.ones((2,4))*2,[True,False],.9),[[1]*4,[2.8]*4])

    def test_both_models_ema_and_resume(self):
        a=small();fill(a)
        old={k:v.copy() for k,v in a.readout.net.p.items()}
        for _ in range(2):a.update()
        self.assertTrue(any(not np.array_equal(old[k],a.readout.net.p[k]) for k in old))
        self.assertTrue(any(not np.array_equal(a.readout.net.p[k],a.readout_target.net.p[k]) for k in old))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.npz';a.save(p);b,_=Learner.load(p)
            self.assertEqual(a.update(),b.update())
            for name in ('flow','target','readout','readout_target'):
                for k in getattr(a,name).net.p:np.testing.assert_array_equal(getattr(a,name).net.p[k],getattr(b,name).net.p[k])
            np.testing.assert_array_equal(a.outcomes([1.,0.],4),b.outcomes([1.,0.],4))
            self.assertEqual(a.action([1.,0.],[1.,2.,3.,4.]),b.action([1.,0.],[1.,2.,3.,4.]))

    def test_shared_bellman_target(self):
        a=small();fill(a);captured={}
        def outcome_train(obs,y,rng,lr):captured['y']=y.copy();return 0.
        def action_train(ctx,actions,rng,lr):captured['ctx']=ctx.copy();return 0.
        a.flow.train=outcome_train;a.readout.train=action_train
        a.update()
        np.testing.assert_array_equal(captured['ctx'][:,2:],captured['y'])

    def test_continuous(self):
        a=small('continuous');fill(a);loss=a.update()
        self.assertTrue(np.isfinite(list(loss.values())).all())
        action=a.action([1.,0.],[1.,2.,3.,4.]);self.assertTrue((action>=-1).all() and (action<=1).all())

    def test_elite_and_ema_inference(self):
        a=small();a.config.elites=2
        a.outcomes=lambda *args:np.array([[0.,0.,0.,0.],[1.,0.,0.,0.],[2.,0.,0.,0.],[3.,0.,0.,0.]])
        np.testing.assert_array_equal(a.choose([1.,0.],reward=[1.,0.,0.,0.]),[2.5,0.,0.,0.])
        a.readout_target.sample=lambda *args:np.array([[-.4,.8]])
        a.readout.sample=lambda *args: (_ for _ in ()).throw(AssertionError('online readout used'))
        self.assertEqual(a.action([1.,0.],[1.,2.,3.,4.]),1)

if __name__=='__main__':unittest.main()
