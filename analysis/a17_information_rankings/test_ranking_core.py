import json
from pathlib import Path
from ranking_core import summarize,tied_ranks
rows=[dict(move_index=i,Q_true=q,receiver_score=s,Qhat=s,delta_effective_dim=s) for i,(s,q) in enumerate([(1.,3.),(2.,2.),(3.,1.)])]
r=summarize(rows,'receiver_score',1e-12);assert r['kendall_tau_b']==-1 and r['spearman']==-1;assert r['top2_overlap']==1 and r['regret_with_no_op']==2 and r['selected_move_index']==2
neg=[dict(move_index=i,Q_true=q,Qhat=s) for i,(s,q) in enumerate([(-1.,2.),(-2.,-3.)])];r2=summarize(neg,'Qhat',1e-12);assert r2['no_op_selected'] and r2['regret_with_no_op']==2
allbad=[dict(move_index=i,Q_true=-i-1.,Qhat=-i-1.) for i in range(3)];r3=summarize(allbad,'Qhat',1e-12);assert r3['regret_with_no_op']==0
constant=[dict(move_index=i,Q_true=q,delta_effective_dim=1.) for i,q in enumerate([2.,-3.,1.])];r4=summarize(constant,'delta_effective_dim',1e-12);assert r4['status']=='CONSTANT_RANKS' and r4['kendall_tau_b'] is None and r4['tie_regret_min']==0 and r4['tie_regret_max']==5
null=[dict(move_index=0,Q_true=1.,Qhat=None),dict(move_index=1,Q_true=None,Qhat=2.)];r5=summarize(null,'Qhat',1e-12);assert r5['status']=='MISSING' and r5['excluded_null_count']==2
near=[dict(move_index=i,Q_true=q,Qhat=s) for i,(s,q) in enumerate([(1.,1.),(1.+1e-13,1.+1e-13),(2.,2.)])];r6=summarize(near,'Qhat',1e-12);assert abs(r6['kendall_tau_b']-1)<1e-14 and len(r6['top2_score_tie_expanded_ids'])==3
within=[dict(move_index=0,Q_true=1e-15,Qhat=1e-15)];r7=summarize(within,'Qhat',1e-12);assert r7['no_op_selected'] and r7['teacher_best_subset_with_no_op']==1e-15
Path(__file__).with_name('TEST_RESULTS.json').write_text(json.dumps(dict(status='PASS',cases=7,scope='independent known ranking / null / constant / numeric ties / no-op examples',records=[r,r2,r3,r4,r5,r6,r7]),indent=2));print('PASS: seven independent ranking cases')
