import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from .core import Config,Learner,bellman_target
from .network import ConditionalUNet


def small(kind='discrete', device='cpu'):
    return Learner(Config(2,4,action_kind=kind,hidden=4,down_dims=(4,6,8),warmup=0,
                         batch_size=4,ode_steps=2,candidates=4,gram_samples=4,gamma=.9), device)

def fill(a):
    for i in range(12):
        action=i%2 if a.config.action_kind=='discrete' else np.array([(-1)**i*.5])
        a.replay.add(np.eye(2)[i%2],action,np.eye(4)[i%4],np.eye(2)[1-i%2])

class Tests(unittest.TestCase):
    def test_coverage_only_changes_readout(self):
        a = small()
        fill(a)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.npz'
            a.save(path)
            b, _ = Learner.load(path, 'cpu')
        uniform = a.update_tensor(coverage=False)
        coverage = b.update_tensor(coverage=True)
        torch.testing.assert_close(uniform[0], coverage[0], rtol=0, atol=0)
        for first, second in ((a.flow.net, b.flow.net), (a.flow.ema, b.flow.ema)):
            for p, q in zip(first.parameters(), second.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(p, q)
                            for p, q in zip(a.readout.net.parameters(), b.readout.net.parameters())))
        self.assertIsNone(a.gram_ema)
        self.assertIsNotNone(b.gram_ema)

    def test_gram_ema_resume(self):
        a = small()
        a.config.gram_ema_rate = .9
        observations = a.tensor([[1., 0.], [0., 1.]])
        first = a.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        second = first * 2

        def gram(targets):
            z = torch.cat((torch.ones_like(targets[:, :1]), targets), dim=1)
            return z[:, :, None] * z[:, None, :] / 2

        a._training_weights(observations, first)
        torch.testing.assert_close(a.gram_ema, gram(first))
        a._training_weights(observations, second)
        expected = .9 * gram(first) + .1 * gram(second)
        torch.testing.assert_close(a.gram_ema, expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.npz'
            a.save(path)
            b, _ = Learner.load(path, 'cpu')
            self.assertEqual(b.config.gram_ema_rate, .9)
            torch.testing.assert_close(b.gram_ema, expected)
            torch.testing.assert_close(a._training_weights(observations, first),
                                       b._training_weights(observations, first), rtol=0, atol=0)
            torch.testing.assert_close(a.gram_ema, .9 * expected + .1 * gram(first))

    def test_outcome_simplex(self):
        devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
        for device in devices:
            a = small(device=device)
            a.config.outcome_simplex = True
            fill(a)
            observations = a.tensor([[1., 0.]] * 4)
            outcomes = a.flow.sample(observations, a.model_rng)
            expected = outcomes.new_full((4,), 1 / (1 - a.config.gamma))
            self.assertTrue((outcomes >= 0).all())
            torch.testing.assert_close(outcomes.sum(1), expected)
            features = a.tensor(np.eye(4))
            targets = bellman_target(features, outcomes, [False] * 4, a.config.gamma)
            torch.testing.assert_close(targets.sum(1), expected)
            self.assertTrue(torch.isfinite(a.update_tensor(coverage=True)).all())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'checkpoint.npz'
                a.save(path)
                b, _ = Learner.load(path, device)
                self.assertTrue(b.config.outcome_simplex)
                np.testing.assert_array_equal(a.outcomes([1., 0.], 4), b.outcomes([1., 0.], 4))

    def test_gradients_network(self):
        generator = torch.Generator().manual_seed(3)
        net = ConditionalUNet(2, 3, 4, (4, 6, 8), generator, torch.float64)
        x, t, c, direction = (torch.rand(shape, generator=generator, dtype=torch.float64)
                              for shape in ((2, 2), (2, 1), (2, 3), (2, 2)))
        (net(x, t, c) * direction).sum().backward()
        parameters = {key: p for key, p, _ in net.checkpoint_parameters()}
        with torch.no_grad():
            for key in ('condition/0/dense/w', 'down/0/0/film/w', 'mid/1/first/ln/scale',
                        'up/0/0/skip/w', 'out/1/b'):
                p = parameters[key]
                idx = (0,) * p.ndim
                orig, eps = p[idx].item(), 1e-5
                p[idx] = orig + eps
                plus = (net(x, t, c) * direction).sum().item()
                p[idx] = orig - eps
                minus = (net(x, t, c) * direction).sum().item()
                p[idx] = orig
                self.assertAlmostEqual(p.grad[idx].item(), (plus - minus) / (2 * eps), places=5, msg=key)

    def test_scaling_and_target(self):
        a=small();y=a.outcomes([1.,0.],16)
        self.assertTrue((y>=0).all() and (y<=10+1e-10).all())
        np.testing.assert_allclose(bellman_target(np.ones((2,4)),np.ones((2,4))*2,[True,False],.9),[[1]*4,[2.8]*4])

    def test_both_models_ema_and_resume(self):
        self.check_ema_and_resume('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_graph_ema_and_resume(self):
        self.check_ema_and_resume('cuda')

    def check_ema_and_resume(self, device):
        a=small(device=device);fill(a)
        old = [p.detach().clone() for p in a.readout.net.parameters()]
        for _ in range(2):a.update()
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(old, a.readout.net.parameters())))
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(a.readout.net.parameters(), a.readout.ema.parameters())))
        a.choose([1., 0.])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.npz';a.save(p);b,_=Learner.load(p, device)
            self.assertEqual(a.update(),b.update())
            for name, (_, net) in a._checkpoint_flows().items():
                for p1, p2 in zip(net.parameters(), b._checkpoint_flows()[name][1].parameters()):
                    torch.testing.assert_close(p1, p2, rtol=0, atol=0)
            np.testing.assert_array_equal(a.outcomes([1.,0.],4),b.outcomes([1.,0.],4))
            np.testing.assert_array_equal(a.choose([1., 0.]), b.choose([1., 0.]))
            self.assertEqual(a.action([1.,0.],[1.,2.,3.,4.]),b.action([1.,0.],[1.,2.,3.,4.]))

    def test_shared_bellman_target(self):
        a = small()
        fill(a)
        captured = {}
        outcome_loss, action_loss = a.flow.loss, a.readout.loss

        def record_outcome(obs, y, rng):
            captured['y'] = y
            return outcome_loss(obs, y, rng)

        def record_action(ctx, actions, rng):
            captured['ctx'] = ctx
            return action_loss(ctx, actions, rng)

        a.flow.loss = record_outcome
        a.readout.loss = record_action
        a.update()
        torch.testing.assert_close(captured['ctx'][:, 2:], captured['y'], rtol=0, atol=0)

    def test_continuous(self):
        a=small('continuous');fill(a);loss=a.update()
        self.assertTrue(np.isfinite(list(loss.values())).all())
        action=a.action([1.,0.],[1.,2.,3.,4.]);self.assertTrue((action>=-1).all() and (action<=1).all())

    def test_elite_and_ema_inference(self):
        a=small();a.config.elites=2
        a.flow.sample=lambda *args:a.tensor([[0.,0.,0.,0.],[1.,0.,0.,0.],[2.,0.,0.,0.],[3.,0.,0.,0.]])
        np.testing.assert_array_equal(a.choose([1.,0.],reward=[1.,0.,0.,0.]),[2.5,0.,0.,0.])
        a.readout.ema.velocity = lambda x, *args: x.new_tensor([[-100., 100.]]).expand_as(x)
        a.readout.net.velocity = lambda *args: self.fail('online readout used')
        self.assertEqual(a.action([1.,0.],[1.,2.,3.,4.]),1)

if __name__=='__main__':unittest.main()
