"""Mock existing compressed context: different QR data rows, same physical r."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from scipy import linalg as la
import json,hashlib,time
import information_diagnostics as d
from info_logger import InfoLogger
OUT=Path(__file__).resolve().parent

def cn(rng,shape):return rng.normal(size=shape)+1j*rng.normal(size=shape)
def relative(a,b):return float(la.norm(a-b)/max(1.,la.norm(a),la.norm(b)))

class MockModel:
    def __init__(self,ctx,U):
        self.U=np.asarray(U,complex); CV=ctx.S@self.U
        self.F=np.stack([self.U.conj().T@b for b in ctx.B])
        R=la.qr(CV,mode='economic')[1]
        self.normal=SimpleNamespace(Z=np.vstack([R@f for f in self.F]))
        Y=CV@self.F
        self.J=np.concatenate([Y.real,Y.imag],axis=1).reshape(-1,ctx.dim_material//2)
        # Complex material input extends physical complex derivative [F, iF].
        C=np.concatenate([Y,1j*Y],axis=2)
        self.J=np.concatenate([C.real,C.imag],axis=1).reshape(-1,ctx.dim_material)
    def j(self,x):return self.J@x
    def jt(self,r):return self.J.T@r

class MockContext:
    def __init__(self,seed):
        rng=np.random.default_rng(seed); self.dim_material=8;self.lam=.01001
        self.S=cn(rng,(5,6));self.B=cn(rng,(2,6,4));self.r=rng.normal(size=20);self.ell=rng.normal(size=8)*.1
        self.builds=0
    def make_model(self,U):self.builds+=1;return MockModel(self,U)

def run():
    started=time.perf_counter();rows=[]
    for seed in range(20261002,20261018):
        ctx=MockContext(seed);Ua=np.eye(6)[:,:2];Ub=np.eye(6)[:,[0,2,3]]
        full=ctx.make_model(np.eye(6)).J; ctx.builds=0
        V,meta=d.common_material_probes(8,'mock_'+str(seed));JV=full@V
        logger=InfoLogger(ctx,fullJ=full,V=V,fullJV=JV,probe_metadata=meta,full_tangent_rhs=meta['actual_count']*2,
                          metric_hash='mock_euclidean',whitening_scope='mock fixed physical real whitened rows',whitening_hash='mock_1')
        x=np.arange(8,dtype=float)*.01
        base=logger.base_summary([0,1],Ua,x,spectrum=True,evaluator=True)
        assert ctx.builds==1 and base['model_cache_hit'] is False
        child=logger.candidate_summary([0,2,3],Ub,x,spectrum=False)
        assert ctx.builds==2 and child['selector_information']['info_effective_dim'] is None
        a,b=logger.model([0,1],Ua),logger.model([0,2,3],Ub)
        assert a.normal.Z.shape[0]!=b.normal.Z.shape[0]
        expected=b.J.T@ctx.r-a.J.T@ctx.r; actual=np.array(child['delta_b'])
        assert relative(expected,actual)<1e-9
        repeat=logger.candidate_summary([0,2,3],Ub,x,spectrum=False)
        assert repeat['model_cache_hit'] and repeat['information_cache_hit'] and ctx.builds==2
        final=logger.final_summary([0,2,3],Ub,x,evaluator=True)
        assert ctx.builds==2 and final['selector_information']['candidate']['weak_evaluated']
        ev=final['evaluator_information'];assert ev['reference_projected']['scope']=='PROJECTED_ONLY'
        assert ev['reference_projected']['full_tangent_rhs']==16
        assert 'full_information_fidelity' not in final['selector_information']
        assert 'reference_projected' not in final['selector_information']
        direct=d.information_fidelity_real(b.J,full)
        assert abs(ev['full_information_fidelity']['info_fidelity_spectral']-direct['info_fidelity_spectral'])<1e-9
        # Same original IDs with genuinely changed U require rebuild (hash-aware cache).
        other=logger.record([0,2,3],np.eye(6)[:,[0,2,4]],x)
        assert ctx.builds==3 and not other['model_cache_hit']
        # Freeze mutation cannot reuse certificates/info under old LM.
        ctx.lam=.003
        try:logger.record([0,1],Ua,x)
        except ValueError:pass
        else:raise AssertionError('logger reused changed frozen context')
        rows.append(dict(seed=seed,delta_b_error=relative(expected,actual),base_compressed_rows=a.normal.Z.shape[0],child_compressed_rows=b.normal.Z.shape[0],physical_real_data_rows=len(ctx.r),builds=ctx.builds,
                         full_reference_fidelity=direct,selector_reference_separation='PASS'))
    out=dict(status='PASS',requested=16,passed=len(rows),rows=rows,wall_s=time.perf_counter()-started,
             scope='mock compressed adapter only; no full Maxwell actions',threshold=1e-9)
    (OUT/'BRIDGE_TEST_RESULTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status=out['status'],passed=len(rows),max_delta_b_error=max(row['delta_b_error'] for row in rows),wall_s=out['wall_s'])))
    return out
if __name__=='__main__':run()
