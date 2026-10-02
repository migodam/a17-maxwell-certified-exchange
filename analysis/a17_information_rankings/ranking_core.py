"""Descriptive paired rankings; numerical ties, not certified intervals."""
import numpy as np
from scipy.stats import rankdata,kendalltau,spearmanr

def tied_ranks(x,tol):
    x=np.asarray(x,float);order=np.argsort(x,kind='stable');labels=np.zeros(len(x),int);group=0;first=None
    for i in order:
        if first is None or x[i]-first>tol:group+=1;first=x[i]
        labels[i]=group
    return rankdata(labels,method='average')

def summarize(rows,metric,tau):
    paired=[r for r in rows if r.get(metric) is not None and r.get('Q_true') is not None]
    out=dict(metric=metric,paired_count=len(paired),excluded_null_count=len(rows)-len(paired),tau_Q=tau,status='MISSING',kendall_tau_b=None,spearman=None,top2_overlap=None,top2_fraction=None,regret_with_no_op=None)
    if not paired:return out
    score=np.array([r[metric] for r in paired]);q=np.array([r['Q_true'] for r in paired]);tol=tau if metric=='Qhat' else float(128*np.finfo(float).eps*max(1.,float(max(abs(score)))));sr=tied_ranks(score,tol);qr=tied_ranks(q,tau)
    out.update(score_tie_tolerance=tol,Q_tie_tolerance=tau,status='DESCRIPTIVE',constant_score=len(set(sr))==1,constant_Q=len(set(qr))==1)
    if len(paired)<2:out.update(status='INSUFFICIENT_PAIRS')
    elif out['constant_score'] or out['constant_Q']:out.update(status='CONSTANT_RANKS',correlation_missing_reason='constant score or Q after declared numerical tie grouping')
    else:out.update(kendall_tau_b=float(kendalltau(sr,qr,variant='b').statistic),spearman=float(spearmanr(sr,qr).statistic))
    # deterministic top2 only; explicitly expose tie-expanded cutoff sets.
    order_s=sorted(range(len(paired)),key=lambda i:(-sr[i],paired[i]['move_index']));order_q=sorted(range(len(paired)),key=lambda i:(-qr[i],paired[i]['move_index']));n=min(2,len(paired));top_s=[paired[i]['move_index'] for i in order_s[:n]];top_q=[paired[i]['move_index'] for i in order_q[:n]]
    boundary_s=sr[order_s[n-1]];boundary_q=qr[order_q[n-1]]
    out.update(top2_score_ids=top_s,top2_Q_ids=top_q,top2_overlap=len(set(top_s)&set(top_q)),top2_fraction=len(set(top_s)&set(top_q))/n,top2_score_tie_expanded_ids=[paired[i]['move_index'] for i in order_s if sr[i]>=boundary_s],top2_Q_tie_expanded_ids=[paired[i]['move_index'] for i in order_q if qr[i]>=boundary_q],top2_scope='candidate-only; numerical ties broken by original move_index; expanded sets also saved')
    best=max(0.,float(max(q)));maxscore=float(max(score));choices=[i for i in range(len(paired)) if sr[i]==max(sr)];chosen=min(choices,key=lambda i:paired[i]['move_index']);no_op=bool(maxscore<=tol);chosen_q=0. if no_op else float(q[chosen]);possible_q=[0.] if no_op else [float(q[i]) for i in choices]
    out.update(teacher_best_subset_with_no_op=best,selected_move_index=None if no_op else paired[chosen]['move_index'],selected_score=0. if no_op else float(score[chosen]),selected_Q_with_no_op=chosen_q,no_op_selected=no_op,regret_with_no_op=best-chosen_q,tied_top_score_count=len(choices),tie_regret_min=best-max(possible_q),tie_regret_max=best-min(possible_q),tie_range_scope='descriptive ambiguity from score ties; not uncertainty interval/certificate',regret_scope='one matched checkpoint, same available paired subset; scores measured relative to no-op zero')
    return out
