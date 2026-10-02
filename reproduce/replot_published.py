"""Replot the final figures from public saved scalar data; no Maxwell actions."""
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(a.data.open()))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    policies=['receiver_no_swap','exact_score_teacher','directed_verified']
    labels=['Receiver backbone','Exact local teacher','Directed verified']
    colors=['#687781','#246A9B','#C9782A']
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.8))
    for oid,ax in zip([2001,2007,2012,2014],axes.flat):
        rr=[r for r in rows if r['figure']=='Figure1' and int(r['object_id'])==oid]
        ax.set_title(rr[0]['panel'],loc='left')
        for policy,label,color in zip(policies,labels,colors):
            points=sorted((int(r['rank']),float(r['value'])) for r in rr if r['policy']==policy and r['metric']=='object_summed_full_GN_gap_three_states')
            assert [x for x,_ in points]==[4,8,16]
            ax.plot(*zip(*points),marker='o',color=color,label=label)
        ax.set_yscale('log');ax.set_xticks([4,8,16]);ax.set_xlabel('Actual shared current rank');ax.set_ylabel('Object-summed full GN gap')
    axes[1,1].legend(fontsize=7);fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(a.out/('Figure1_object_GN_gap.'+ext),dpi=250)
    plt.close(fig)
    rr=[r for r in rows if r['figure']=='Figure2']
    values={r['metric']:float(r['value']) for r in rr}
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.8))
    for ax,key,title,ylabel in zip(axes.flat[:3],['delta_logdet_volume','delta_effective_dim','Q_true'],['Information volume','Effective information','Positive information, negative utility'],['Change in log-det volume','Change in effective dimension','Actual full-quadratic gain']):
        ax.bar([0],[values[key]],color='#246A9B' if key!='Q_true' else '#C9782A')
        ax.axhline(0,color='#687781',lw=.7);ax.set_xticks([0],['Same current exchange']);ax.set_title(title,loc='left');ax.set_ylabel(ylabel)
    ax=axes[1,1];ax.bar(range(3),[values[x] for x in ['Q_A','Q_B','Q_pair']],color=['#687781','#687781','#246A9B'])
    ax.axhline(0,color='#687781',lw=.7);ax.set_xticks(range(3),['Single A\nrank 15','Single B\nrank 15','Pair A+B\nrank 16']);ax.set_title('Coupled additions',loc='left');ax.set_ylabel('Gain from the same rank-14 core')
    fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(a.out/('Figure2_information_and_pair_witnesses.'+ext),dpi=250)
    plt.close(fig)
    print('PASS: figures redrawn from saved scalars; no new physical experiment')

if __name__=='__main__':main()
