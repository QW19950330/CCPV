"""Tiny manufactured fixtures only; no scientific input or training runs."""
import unittest
import numpy as np
import torch
import matched_model as M

torch.set_num_threads(1)


def fixture():
    rng=np.random.default_rng(13)
    initial={k:rng.normal(0,.12,size=shape).astype(np.float64) for k,shape in zip(M.STATE,M.SHAPES)}
    x=torch.from_numpy(rng.normal(size=(3,16,72)))
    y=torch.tensor([[0.,0.],[1.,0.],[1.,1.]],dtype=torch.float64)
    w=torch.tensor([.2,.3,.5],dtype=torch.float64)
    return M.make(initial),x,y,w


def gradients(model,objective):
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(objective,list(model.parameters()),retain_graph=True)]).detach().numpy()


class Handmade(unittest.TestCase):
    def test_K1_and_identical16_match_original_point_loss_and_gradient(self):
        model,x,y,w=fixture(); one=x[:,:1]; same=one.repeat(1,16,1)
        point=model(one[:,0]); losses=torch.nn.functional.binary_cross_entropy_with_logits(point,y,reduction='none')
        old=(w[:,None]*losses).sum(0).sum()+M.RIDGE*sum(p.square().sum() for n,p in model.named_parameters() if n.endswith('weight'))
        base=gradients(model,old)
        for data in (one,same):
            obj,bce,reg,lp,ln=M.terms(model,data,y,w)
            self.assertAlmostEqual(float(obj),float(old),places=13)
            np.testing.assert_allclose(gradients(model,obj),base,atol=3e-15,rtol=3e-13)
            np.testing.assert_allclose(torch.logaddexp(lp,ln).detach().numpy(),0.,atol=4e-16)

    def test_nonidentical_finite_difference_all_parameter_blocks(self):
        model,x,y,w=fixture(); obj=M.terms(model,x,y,w)[0]
        analytic=torch.autograd.grad(obj,list(model.parameters()))
        for p,g in zip(model.parameters(),analytic):
            for i in (0,p.numel()-1):
                flat=p.view(-1); original=float(flat[i]); eps=1e-6
                with torch.no_grad(): flat[i]=original+eps
                plus=float(M.terms(model,x,y,w)[0])
                with torch.no_grad(): flat[i]=original-eps
                minus=float(M.terms(model,x,y,w)[0])
                with torch.no_grad(): flat[i]=original
                self.assertAlmostEqual((plus-minus)/(2*eps),float(g.reshape(-1)[i]),delta=5e-9)

    def test_probability_average_is_not_average_particle_loss(self):
        model,x,y,w=fixture(); obj,bce,reg,lp,ln=M.terms(model,x,y,w)
        eta=model(x.reshape(-1,72)).reshape(3,16,2)
        avg_loss=(w[:,None]*torch.nn.functional.binary_cross_entropy_with_logits(eta,y[:,None,:].repeat(1,16,1),reduction='none').mean(1)).sum()
        self.assertLess(float(bce.sum()),float(avg_loss)-1e-7)
        meanprob=eta.sigmoid().mean(1)
        literal=(w[:,None]*-(y*meanprob.log()+(1-y)*torch.log1p(-meanprob))).sum()+reg
        self.assertAlmostEqual(float(obj),float(literal),places=13)

    def test_duplicate_particles_do_not_multiply_anchor_weight_ridge_or_clip(self):
        model,x,y,w=fixture(); obj,bce,reg,*_=M.terms(model,x,y,w)
        doubled=M.terms(model,x.repeat_interleave(2,dim=1),y,w)
        np.testing.assert_allclose(float(obj),float(doubled[0]),atol=4e-16)
        np.testing.assert_allclose(gradients(model,obj),gradients(model,doubled[0]),atol=3e-15,rtol=3e-13)
        self.assertEqual(float(reg),float(doubled[2]))
        obj.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.01,error_if_nonfinite=True)
        raw=gradients(model,M.terms(model,x,y,w)[0]); clipped=np.concatenate([p.grad.numpy().ravel() for p in model.parameters()])
        np.testing.assert_allclose(clipped,raw*min(1.,.01/(float(norm)+1e-6)),rtol=1e-12,atol=1e-15)

    def test_saved_initial_and_frozen_missing_feature_rules(self):
        model,x,y,w=fixture(); initial=M.state(model); new=M.make(initial)
        self.assertTrue(all(np.array_equal(initial[k],M.state(new)[k]) for k in M.STATE))
        norm=dict(mean=np.arange(36,dtype=float),scale=np.ones(36),completely_absent_training_feature=np.zeros(36,bool))
        norm['completely_absent_training_feature'][1]=True
        a=np.arange(36,dtype=float)[None,:].repeat(2,0);a[0,0]=np.nan;a[1,1]=999
        z=M.transform(a,norm)
        self.assertEqual(z.shape,(2,72)); self.assertEqual(z[0,0],0);self.assertEqual(z[0,36],1)
        self.assertEqual(z[1,1],0);self.assertEqual(z[1,37],0)

    def test_extreme_logits_finite_mixture(self):
        model,x,y,w=fixture()
        with torch.no_grad(): model.output.weight.zero_();model.output.bias.copy_(torch.tensor([1000.,-1000.]))
        obj,*_=M.terms(model,x,y,w)
        self.assertTrue(np.isfinite(float(obj)));self.assertTrue(np.isfinite(gradients(model,obj)).all())


if __name__=='__main__': unittest.main(verbosity=2)
