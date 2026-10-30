"""Two offline presentation figures from synchronized, already-recorded results.

No model import, NRMSE calculation, ledger recalculation or remote access.
Run only after parent syncs terminal data:
  python plot_extra_controls.py --root /absolute/study/root
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

CASES=('digit3','letter_e','torus','double_torus','cross3d','offset_shell')
POLICIES=('full_gn','receiver','projected_internal','fourier','random','exchange')
LABELS={'full_gn':'Full-GN proposal + common safeguard\nfull current; unrestricted k',
        'receiver':'SOM-like receiver proxy | k=16',
        'projected_internal':'TSOM-style internal proxy | k=16',
        'fourier':'Fourier proposal | k=16','random':'Random proposal | k=16',
        'exchange':'A17 exchange proposal | k=16',
        'full_gradient':'Full-gradient safeguard-only control\nunrestricted k; posthoc'}
COLORS={p:plt.get_cmap('tab10').colors[i] for i,p in enumerate(POLICIES)}


def number(value):
    if value in (None,''):return None
    return float(value)


def recorded_status(raw):
    status=str(raw or 'STATUS UNKNOWN')
    if status=='MAX_UPDATES' or 'BUDGET' in status:return 'budget-capped ['+status+']'
    return status


def cost_error_scatter(root,out):
    # The CSV is the sole source of both axes. Do not substitute solver counters
    # for admitted/accounted RHS, or calculate errors from material arrays.
    with (root/'analysis'/'FINAL_METRICS.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    points=[];missing=[]
    for case in CASES:
        for policy in POLICIES:
            rec=next((r for r in rows if r.get('case')==case and r.get('policy')==policy),None)
            x=number(rec.get('accounted_complex_full_rhs')) if rec else None
            y=number(rec.get('complex_nrmse')) if rec else None
            if x is None or y is None:
                missing.append((case,policy));continue
            points.append((case,policy,x,y,rec))
    fig,axes=plt.subplots(2,3,figsize=(16,10),sharex=True,sharey=True,layout='constrained')
    xmax=max([1.]+[p[2] for p in points]);ymax=max([1.]+[p[3] for p in points])
    for ax,case in zip(axes.ravel(),CASES):
        for idx,policy in enumerate(POLICIES):
            point=next((p for p in points if p[0]==case and p[1]==policy),None)
            if point is None:continue
            _,_,x,y,rec=point
            ax.scatter([x],[y],c=[COLORS[policy]],s=75,marker='o',edgecolors='white',linewidth=.7,zorder=3)
            ax.annotate(f'{y:.3f}',(x,y),xytext=(5,5 if idx%2==0 else -12),textcoords='offset points',fontsize=8,color=COLORS[policy])
        absent=[p for c,p in missing if c==case]
        if absent:ax.text(.02,.98,'Missing recorded cost/error: '+', '.join(absent),transform=ax.transAxes,va='top',fontsize=7,color='firebrick',wrap=True)
        ax.set_title(case);ax.grid(alpha=.22);ax.set_xlim(-.025*xmax,1.10*xmax);ax.set_ylim(0,1.08*ymax)
        ax.set_xlabel('Accounted / admitted complex full RHS');ax.set_ylabel('Recorded final complex NRMSE (lower is better)')
    handles=[Line2D([],[],color=COLORS[p],marker='o',linestyle='',label=LABELS[p]) for p in POLICIES]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=8,frameon=False)
    fig.suptitle('Primary six methods on six objects | recorded error versus admitted RHS cost\nAdmitted RHS is not matched wall time. All primary methods share the full-gradient safeguard.\nFull-GN uses full current; five current-space methods use k=16. Gradient-only controls appear separately.',fontsize=12)
    fig.savefig(out/'RECORDED_NRMSE_VS_ADMITTED_RHS.png',dpi=160);plt.close(fig)


def cube_from_saved(points,values):
    # Reindex saved native voxels; no interpolation or new physical evaluation.
    axes=[np.unique(points[:,i]) for i in range(3)]
    cube=np.empty(tuple(len(a) for a in axes),dtype=np.asarray(values).dtype)
    ids=tuple(np.searchsorted(axes[i],points[:,i]) for i in range(3))
    cube[ids]=values
    return cube,axes


def control_slices(root,out):
    columns=('truth','full_gn','receiver','exchange','full_gradient');scenes={}
    for case in ('digit3','torus'):
        input_path=root/'inputs'/f'{case}.npz'
        with np.load(input_path,allow_pickle=False) as data:
            input_points=data['points'].copy();truth=data['truth'].copy()
        cube,axes=cube_from_saved(input_points,truth)
        scenes[case]={'truth':dict(cube=cube,axes=axes,status='SYNTHETIC TRUTH')}
        for policy in columns[1:]:
            folder=(root/'gradient_control'/case if policy=='full_gradient' else root/'results'/case/policy)
            rec={}
            if (folder/'final.json').exists():
                meta=json.loads((folder/'final.json').read_text());rec.update(status=recorded_status(meta.get('status')),
                    metrics=meta.get('metrics') or {},updates=meta.get('updates'))
            else:rec['status']='NOT_RUN / final.json missing'
            if (folder/'final.npz').exists():
                with np.load(folder/'final.npz',allow_pickle=False) as data:
                    points=data['points'] if 'points' in data else input_points
                    cube,axes=cube_from_saved(points,data['chi'])
                rec.update(cube=cube,axes=axes)
            scenes[case][policy]=rec
    reals=[rec['cube'].real for scene in scenes.values() for rec in scene.values() if 'cube' in rec]
    low=min([0.]+[float(c.min()) for c in reals]);high=max([1.5]+[float(c.max()) for c in reals])
    norm=Normalize(low,high,clip=False);cmap='viridis'
    fig,axes=plt.subplots(2,5,figsize=(19,9),layout='constrained')
    for row,case in enumerate(('digit3','torus')):
        for col,policy in enumerate(columns):
            ax=axes[row,col];rec=scenes[case][policy]
            if policy=='truth':caption=case+' | SYNTHETIC TRUTH\nNOT RECONSTRUCTION'
            else:
                value=number(rec.get('metrics',{}).get('complex_nrmse'))
                score=f'{value:.3f}' if value is not None else 'unavailable'
                count=rec.get('updates');updates=str(count) if count is not None else 'unavailable'
                caption=LABELS[policy]+f'\nNRMSE={score} | accepted updates={updates}\n'+rec['status']
            ax.set_title(caption,fontsize=8)
            if 'cube' in rec:
                layer=int(np.argmin(np.abs(rec['axes'][2])));image=rec['cube'][:,:,layer].real
                ax.imshow(image.T,origin='lower',extent=(-.5,.5,-.5,.5),interpolation='nearest',cmap=cmap,norm=norm)
                ax.text(.02,.02,f'z={rec["axes"][2][layer]:+.5f}',transform=ax.transAxes,fontsize=8,color='white',bbox={'facecolor':'black','alpha':.5,'pad':1})
            else:ax.text(.04,.45,'NO SAVED CONTROL/FINAL ARRAY\n'+rec['status'],transform=ax.transAxes,color='firebrick',fontsize=8,wrap=True)
            ax.set_xlabel('x');ax.set_ylabel('y');ax.set_xticks([-.5,0,.5]);ax.set_yticks([-.5,0,.5])
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axes.ravel().tolist(),shrink=.65,label=f'Re chi | common native-array range [{low:.4g}, {high:.4g}]')
    fig.suptitle('Native central XY Re chi | truth, main final endpoints, and separate full-gradient control\nRecorded NRMSE and accepted counts only. Nearest pixels; no smoothing. Controls are posthoc with different histories.',fontsize=12)
    fig.savefig(out/'NATIVE_XY_MAIN_AND_FULL_GRADIENT_CONTROL.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    cost_error_scatter(root,out);control_slices(root,out)


if __name__=='__main__':main()
