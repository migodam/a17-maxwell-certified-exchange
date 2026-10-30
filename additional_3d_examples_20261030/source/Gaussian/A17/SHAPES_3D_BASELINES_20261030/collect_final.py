"""Collect already-recorded outcomes; no re-solving or repeated numeric audits."""
import csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'CONFIG.json').read_text())

def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default

def main():
    rows=[];objects={};charges={}
    for line in (ROOT/'COST_LEDGER.jsonl').read_text().splitlines():
        event=json.loads(line);parts=event.get('stage','').split('/')
        key=tuple(parts[:2]);entry=charges.setdefault(key,dict(state=0,rhs=0))
        entry[event['kind']]+=event['count']
    for case in CFG['cases']:
        obj={};objects[case]=obj
        for policy in CFG['policies']:
            folder=ROOT/'results'/case/policy;f=read(folder/'final.json')
            if f is None:
                rows.append(dict(case=case,policy=policy,status='NOT_RUN'));continue
            tr=read(folder/'trajectory.json',[]);m=f.get('metrics') or {}
            row=dict(case=case,policy=policy,status=f['status'],updates=f['updates'],attempted_updates=f.get('attempted_updates'),wall_s=f['wall_s'],gradient_fallback_count=sum(bool(t.get('gradient_fallback')) for t in tr),accepted_gradient_fallback_count=sum(bool(t.get('gradient_fallback')) and bool(t.get('accepted')) for t in tr),accepted_policy_proposal_count=sum(not bool(t.get('gradient_fallback')) and bool(t.get('accepted')) for t in tr),**m)
            counters=(f.get('counters') or {}).get('totals',{})
            row['full_factorizations']=counters.get('lu_factorizations')
            row['full_tangent_rhs']=counters.get('solve_rhs_full_tangent',0)
            row['full_adjoint_rhs']=counters.get('solve_rhs_full_adjoint',0)
            row['full_forward_rhs']=counters.get('solve_rhs_full_forward',0)
            row['counter_scope']=f.get('counters_scope','whole trajectory')
            row['continued_from_iteration']=(f.get('continuation') or {}).get('next_iteration')
            admitted=charges.get((case,policy),dict(state=0,rhs=0))
            row['accounted_full_factorization_attempts']=admitted['state']
            row['accounted_complex_full_rhs']=admitted['rhs']
            rows.append(row);obj[policy]=row
        if 'exchange' in obj:
            e=obj['exchange']['complex_nrmse']
            obj['A17_ratios']={p:e/obj[p]['complex_nrmse'] for p in CFG['policies'] if p!='exchange' and p in obj}
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'FINAL_METRICS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    controls={case:read(ROOT/'gradient_control'/case/'final.json') for case in ('digit3','torus')}
    result=dict(analysis='recorded final metrics and flags only; no repeated numerical comparison or solver',objects=objects,rows=rows,cost=read(ROOT/'COST_STATE.json'),execution_config=read(ROOT/'CONFIG_EXTENDED.json',CFG),terminal=read(ROOT/'RUN_TERMINAL.json'),gradient_control=controls)
    (out/'FINAL_SUMMARY.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(final_records=sum(r['status']!='NOT_RUN' for r in rows),expected=36,recorded_statuses={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})})))

if __name__=='__main__':main()
