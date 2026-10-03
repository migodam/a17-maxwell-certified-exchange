"""Independent CPU algebra/API tests; no physical kernels or reference labels."""
import importlib.util,inspect,pathlib,unittest
import numpy as np
P=pathlib.Path(__file__).resolve().parents[1]/'code/online_tolerance.py'
spec=importlib.util.spec_from_file_location('online_tolerance_under_test',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class OnlineToleranceTests(unittest.TestCase):
 def inputs(self,complex_output=False):
  rng=np.random.default_rng(20261002);p=7;n=11;c=3
  J=rng.normal(size=(n,p));r=rng.normal(size=n)
  if complex_output:J=J+1j*rng.normal(size=J.shape);r=r+1j*rng.normal(size=n)
  x=rng.normal(size=p);d=rng.normal(size=(p,c));ell=rng.normal(size=p);lam=.03
  return J,r,x,d,ell,lam
 def test_real_quadratic_objective_difference(self):self.check_objective(False)
 def test_complex_output_quadratic_objective_difference(self):self.check_objective(True)
 def check_objective(self,c):
  J,r,x,d,ell,lam=self.inputs(c);u=r+J@x;z=J@d
  # Real r argument supplies output dimension/energy only for complex-output API test.
  a=m.directional_check(r.real,u,z,x,d,ell,lam)
  def phi(t):return .5*np.vdot(r+J@t,r+J@t).real+ell@t+.5*lam*(t@t)
  diff=np.array([phi(x)-phi(x+d[:,i]) for i in range(d.shape[1])]);err=np.abs(a['gain']-diff)
  self.assertTrue(np.all(err<=a['tau']),repr((err,a['tau'])))
  np.testing.assert_allclose(a['gain'],-a['observation_term']-a['prior_linear_term']-a['data_curvature_cost']-a['prior_curvature_cost'],rtol=0,atol=0)
 def test_complex_and_realified_output_gain_agree(self):
  J,r,x,d,ell,lam=self.inputs(True);u=r+J@x;z=J@d
  a=m.directional_check(r.real,u,z,x,d,ell,lam);b=m.directional_check(np.r_[r.real,r.imag],np.r_[u.real,u.imag],np.vstack([z.real,z.imag]),x,d,ell,lam)
  self.assertTrue(np.all(np.abs(a['gain']-b['gain'])<=np.maximum(a['tau'],b['tau'])))
 def test_single_direction_and_column_agree(self):
  J,r,x,d,ell,lam=self.inputs();a=m.directional_check(r,r+J@x,J@d[:,0],x,d[:,0],ell,lam);b=m.directional_check(r,r+J@x,J@d[:,:1],x,d[:,:1],ell,lam)
  np.testing.assert_array_equal(a['gain'],b['gain']);np.testing.assert_array_equal(a['tau'],b['tau'])
 def test_material_complex_rejected(self):
  J,r,x,d,ell,lam=self.inputs()
  for v in ('x','d','ell'):
   args=dict(r=r,u=r+J@x,z=J@d,x=x,d=d,ell=ell,lam=lam);args[v]=args[v].astype(complex)
   with self.subTest(v=v),self.assertRaises(ValueError):m.directional_check(**args)
 def test_nonfinite_and_nonpositive_lambda_rejected(self):
  J,r,x,d,ell,lam=self.inputs();args=dict(r=r,u=r+J@x,z=J@d,x=x,d=d,ell=ell,lam=lam)
  for v in args:
   for bad in (np.nan,np.inf):
    a={k:np.array(y,copy=True) if k!='lam' else y for k,y in args.items()}
    if v=='lam':a[v]=bad
    else:a[v].flat[0]=bad
    with self.subTest(v=v,bad=bad),self.assertRaises(ValueError):m.directional_check(**a)
  for v in (0.,-1.):
   with self.assertRaises(ValueError):m.directional_check(**{**args,'lam':v})
 def test_shape_rejection(self):
  J,r,x,d,ell,lam=self.inputs();a=dict(r=r,u=r+J@x,z=J@d,x=x,d=d,ell=ell,lam=lam)
  with self.assertRaises(ValueError):m.directional_check(**{**a,'r':r.astype(complex)})
  cases={'r':r[:,None],'u':np.zeros(len(r)+1),'x':x[:,None],'ell':ell[:-1],'d':d[:-1], 'z':np.zeros((len(r),d.shape[1]+1))}
  for k,v in cases.items():
   with self.subTest(field=k),self.assertRaises(ValueError):m.directional_check(**{**a,k:v})
 def test_scalar_direction_rejected_cleanly(self):
  J,r,x,d,ell,lam=self.inputs()
  with self.assertRaises(ValueError):m.directional_check(r,r+J@x,0.,x,0.,ell,lam)
 def test_dimension_gamma_rule(self):
  vals=[]
  for n,p in ((3,2),(31,7),(1536,54)):
   a=m.directional_check(np.ones(n),np.ones(n),np.zeros(n),np.zeros(p),np.zeros(p),np.zeros(p),1.)
   N=n+3*p+16;e=np.finfo(float).eps;self.assertEqual(a['accumulation_length'],N);self.assertEqual(a['tau_rel'],m.SAFETY*(N*e)/(1-N*e));vals.append(a['tau_rel'])
  self.assertTrue(vals[0]<vals[1]<vals[2])
 def test_positive_negative_nearzero_and_noop(self):
  x=np.zeros(1);ell=np.zeros(1);r=np.ones(1)
  for z,sgn in ((-.1,1),(.1,-1),(-1e-20,1),(1e-20,-1)):
   a=m.directional_check(r,r,np.array([z]),x,np.zeros(1),ell,1.)
   self.assertEqual(np.sign(a['gain'][0]),sgn)
   self.assertEqual(bool(a['gain'][0]>a['tau'][0]),sgn>0 and abs(z)>.01)
  a=m.directional_check(r,r,np.zeros(1),x,np.zeros(1),ell,1.);self.assertEqual(a['gain'][0],0.);self.assertGreater(a['tau'][0],0.);self.assertFalse(a['gain'][0]>a['tau'][0])
 def test_api_has_only_online_inputs(self):
  self.assertEqual(list(inspect.signature(m.directional_check).parameters),['r','u','z','x','d','ell','lam'])
  a=m.directional_check(np.ones(1),np.ones(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),1.)
  self.assertFalse(a['deterministic_certificate']);self.assertFalse(any(k in a for k in ('fullref','floor','truth','teacher','evaluator')))

if __name__=='__main__':unittest.main(verbosity=2)
