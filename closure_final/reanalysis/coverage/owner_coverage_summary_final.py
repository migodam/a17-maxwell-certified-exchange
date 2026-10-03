"""Saved-only owner statistics; no Maxwell actions or online rule changes."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
EVIDENCE = WORKSPACE / 'research/delegated/a17_closure_collectors_v5/real_saved_audit_v5/coverage_evidence.json'
EXPECTED = 'e4f7c13169e570d49d6b5e0e16e8111c9c94e891dd419cded826c2652c40d978'
def main():
    raw = EVIDENCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    e = json.loads(raw)
    complete = [r for r in e['object_initial_opportunity'] if r['all_planned_initial_complete']]
    assert len(complete) == 10
    cells = [r for r in e['cells'] if r['binding_status'] == 'VERIFIED']
    assert len(cells) == 96
    stats = {}
    for key in ['anchor_top1_significance_verified_noop_capture', 'balanced12_capture']:
        a = np.array([r[key] for r in complete])
        stats[key] = dict(min=float(a.min()), max=float(a.max()), median=float(np.median(a)),
                          iqr=[float(x) for x in np.quantile(a, [.25,.75])])
    path = dict(online_better=0, teacher_better=0, within_evaluation_floor=0)
    path_objects = []
    for oid in sorted({r['object_id'] for r in cells}):
        rows = [r for r in cells if r['object_id'] == oid]
        online = sum(r['actual_online_final_full_gap'] for r in rows)
        teacher = sum(r['local_teacher_final_full_gap'] for r in rows)
        path_objects.append(dict(object_id=oid, observed_cells=len(rows), complete=len(rows)==9,
                                 online_sum=online, bounded_teacher_sum=teacher,
                                 online_to_bounded_teacher_ratio=online/teacher,
                                 partial_scope='9 planned cells; missing cells not imputed'))
        for r in rows:
            native = json.loads((ROOT/'remote_snapshots/closure_final_closed_v1/source'/r['directory']/f"k{r['k']}"/'result.json').read_text())
            tol = max(native['actual_online_final_risk']['numerical_floor'],native['teacher_final_risk']['numerical_floor'])
            diff = r['actual_minus_local_teacher_risk']
            path['teacher_better' if diff>tol else 'online_better' if diff < -tol else 'within_evaluation_floor'] += 1
    result = dict(schema='a17.owner.coverage.statistics.final.v1', evidence_sha256=EXPECTED,
                  scope='INITIAL_1SWAP_OPPORTUNITY_AND_BOUNDED_LOCAL_3ACTION_PATH_NOT_GLOBAL_ORACLE',
                  complete_objects=10, partial_objects=1, actual_cells=96, planned_cells=99,
                  initial_object_statistics=stats, initial_object_rows=e['object_initial_opportunity'],
                  path_comparison=path, path_object_rows=path_objects,
                  limitations=['Initial positive-Q weighted captures are not deployed multi-step capture.',
                               'Path differences combine candidate coverage, ranking, and search interactions.',
                               'The online rule can outperform the greedy full-dictionary path; it is not a global oracle.',
                               'Evaluation-only floors used here do not enter online acceptance.'])
    out=ROOT/'analysis/OWNER_COVERAGE_STATISTICS_FINAL.json'
    assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['complete_objects','actual_cells','planned_cells','initial_object_statistics','path_comparison']},indent=2))
if __name__ == '__main__': main()
