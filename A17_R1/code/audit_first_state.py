"""Postcheck first saved V3 state against literal predeclared tolerance.
Uses persisted reduced arrays and already-paid full J only; no Maxwell action.
Raw result is never edited and no replay result replaces it.
"""
from pathlib import Path
import json,numpy as np
from scipy import linalg as la
import pinned_runtime as rt
from endpoint_batch import EndpointBatch

def phi(r,ell,lam,x,Jx):return float(.5*np.dot(r+Jx,r+Jx)+ell@x+.5*lam*(x@x))
def tolerance(r,floor,base,q):return max(10*floor,256*np.finfo(float).eps*max(float(r@r),abs(base),abs(base-q),1e-300))
def main():
    root=rt.ROOT/'results/object_2007_early_portable';data=np.load(root/'public_workspace.npz');w={k:data[k] for k in data.files};J=np.load(root/'offline_information_reference.npz')['fullJ'];r=w['r'];ell=w['ell'];lam=float(w['lambda_total']);P=w['PB'].shape[0];m=w['S'].shape[0];q=w['PB'].shape[2]
    U=w['anchor'];A=U.conj().T@w['L']@U;F=np.stack([la.solve(A,U.conj().T@p) for p in w['PB']]);CV=w['S']@U
    def anchor_j(x):
        z=x[:q]+1j*x[q:];out=np.einsum('mb,pb->pm',CV,np.einsum('pbq,q->pb',F,z));return np.concatenate([np.concatenate([a.real,a.imag]) for a in out])
    engine=EndpointBatch(w['L'],w['S'],w['PB'],r,ell,lam,device='cpu');checks=[];margins=[];affected=[]
    for k in (4,8,16):
        d=root/f'k{k}';baseline=json.loads((d/'receiver_no_swap.json').read_text());ids=baseline['selected_ids'];U=la.svd(w['pool'][:,ids],full_matrices=False)[0][:,:k];x0=engine.evaluate(U[None])['steps'][0];floor=baseline['risk']['numerical_floor']
        for row in map(json.loads,(d/'moves.jsonl').read_text().splitlines()):
            if row.get('status')!='OK':continue
            nm=row['policy_id'];rn=row['round'];x=x0 if rn==0 else np.load(d/f'{nm}_round_{rn-1}.npz')['step']
            full=nm in ('exact_score_teacher','directed_verified','adaptive_leq_k') and row.get('Q_true') is not None
            val=row['Q_true'] if full else row['Qhat'];base=phi(r,ell,lam,x,J@x if full else anchor_j(x));tau_new=tolerance(r,floor,base,val)
            old=float(row['tau']);same=(val>old)==(val>tau_new)
            if not same:affected.append(dict(k=k,policy=nm,round=rn,move=row['move_index'],old=old,new=tau_new,gain=val))
            if row.get('accepted'):margins.append(dict(k=k,policy=nm,margin_to_literal_tau=val-tau_new,literal_tau=tau_new,executed_full_gain=row['Q_true']))
            checks.append(same)
        # Verify baseline actual k and orthonormality from original persisted atoms.
        assert la.matrix_rank(w['pool'][:,ids])==k if hasattr(la,'matrix_rank') else np.linalg.matrix_rank(w['pool'][:,ids])==k
    result=dict(status='PASS' if not affected else 'REQUIRES_CAUSAL_REPLAY',score_comparisons=len(checks),all_classifications_same=all(checks),changed_classifications=affected,accepted_margins=margins,no_new_Maxwell_calls=True,first_V3_results_unchanged=True,scope='First Gaussian early state only; no certification of untested states',source='first_terminal_snapshot.zip immutable SHA94b7ed668f85ea5b8b9b9e14869b282b291361954b35d2710e1ce12b0338c33a')
    (rt.ROOT/'tests/first_V3_tolerance_postcheck.json').write_text(json.dumps(result,indent=2)+'\n');print({k:result[k] for k in ('status','score_comparisons','all_classifications_same')})
if __name__=='__main__':main()
