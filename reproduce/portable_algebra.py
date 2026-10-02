"""New independent portable algebra regression; not historical Maxwell test."""
import numpy as np

def check():
 rng=np.random.default_rng(20261002)
 J=rng.normal(size=(9,4));D=rng.normal(size=(9,4));r=rng.normal(size=9);lam=.3
 H=J.T @ J+lam*np.eye(4);Jn=J+D;Hn=Jn.T @ Jn+lam*np.eye(4)
 old=-np.linalg.solve(H,J.T @ r);new=-np.linalg.solve(Hn,Jn.T @ r)
 db=D.T @ r;dI=J.T @ D+D.T @ J+D.T @ D
 predicted=-np.linalg.solve(Hn,db+dI @ old)
 np.testing.assert_allclose(new-old,predicted,rtol=1e-11,atol=1e-12)
 objective=lambda x:.5*np.linalg.norm(r+J @ x)**2+.5*lam*(x @ x)
 delta=rng.normal(size=4);u=r+J @ old;z=J @ delta
 gain=-(u @ z)-lam*(old @ delta)-.5*((z @ z)+lam*(delta @ delta))
 np.testing.assert_allclose(objective(old)-objective(old+delta),gain,rtol=1e-11,atol=1e-12)
 return {'status':'PASS_INDEPENDENT_CPU_ALGEBRA_ONLY','scope':'shared real variables, information/RHS paired step and directional scalar gain; no Maxwell, reconstruction, certificate or novelty claim'}
if __name__=='__main__':print(check())
