"""Offline saved-array plots only; no Maxwell, old-results or runner imports.

Run: python plot_results.py --root /absolute/new/experiment/root
Only root/inputs and root/results are read. Missing/failed panels stay visible.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shapes import TARGETS

POLICIES = ('full_gn', 'receiver', 'projected_internal', 'fourier', 'random', 'exchange')
LABELS = {'truth': 'SYNTHETIC TRUTH', 'full_gn': 'fullGN+safeguard',
          'receiver': 'receiver (SOM-like)', 'projected_internal': 'internal (TSOM-style proxy)',
          'fourier': 'Fourier', 'random': 'Random', 'exchange': 'A17 exchange'}
FIGURE_KEY = ('18-update cap | n = accepted updates; G = gradient-safeguard attempts / total attempts\n'
              'same-grid n12 | fullGN: full current; other methods: k=16')
THRESHOLD = .5


def grid(points, values):
    p, v = np.asarray(points), np.asarray(values)
    if p.ndim != 2 or p.shape[1] != 3 or v.shape != (len(p),):
        raise ValueError('Expected points(N,3) and chi(N,)')
    if not np.isfinite(p).all() or not np.isfinite(v).all():
        raise ValueError('Non-finite saved points/chi')
    axes = [np.unique(p[:, i]) for i in range(3)]
    shape = tuple(len(a) for a in axes)
    if len(set(shape)) != 1 or np.prod(shape) != len(p):
        raise ValueError('Expected complete cubic Cartesian grid')
    ids = tuple(np.searchsorted(axes[i], p[:, i]) for i in range(3))
    if len(np.unique(np.ravel_multi_index(ids, shape))) != len(p):
        raise ValueError('Repeated grid points')
    cube = np.empty(shape, complex)
    cube[ids] = v
    edges = []
    for a in axes:
        if len(a) < 2 or not np.allclose(np.diff(a), np.diff(a)[0]):
            raise ValueError('Expected uniform grid')
        h = a[1]-a[0]
        e = np.r_[a[0]-h/2, (a[:-1]+a[1:])/2, a[-1]+h/2]
        if not np.allclose(e[[0, -1]], [-.5, .5], atol=1e-8):
            raise ValueError('Grid domain differs from [-.5,.5]^3')
        edges.append(e)
    return cube, axes, edges


def status_label(value):
    raw = str(value or 'STATUS_UNKNOWN')
    if raw.upper() in ('MAX_UPDATES', 'MAX_ITER', 'MAXITER', 'MAX_ITERATIONS') or 'BUDGET' in raw.upper():
        return f'budget-capped [{raw}]'
    return raw


def read(root):
    scenes, receipt = {}, {'root': str(root), 'physics_actions': 0, 'threshold': THRESHOLD,
                           'method_order': list(POLICIES), 'records': []}
    for name in TARGETS:
        inp = root/'inputs'/f'{name}.npz'
        scene = {}
        points = None
        try:
            with np.load(inp, allow_pickle=False) as data:
                points = data['points'].copy()
                cube, axes, edges = grid(points, data['truth'])
            scene['truth'] = dict(cube=cube, axes=axes, edges=edges, status='SYNTHETIC TRUTH')
        except (OSError, ValueError, KeyError) as exc:
            scene['truth'] = dict(status=f'MISSING/INVALID INPUT: {exc}')
        for policy in POLICIES:
            folder = root/'results'/name/policy
            rec = dict(status='NOT_RUN / final missing')
            try:
                meta = json.loads((folder/'final.json').read_text())
                rec['status'] = status_label(meta.get('status'))
                rec['metrics'] = meta.get('metrics') or {}
                rec['updates'] = meta.get('updates')
                recorded_rows=json.loads((folder/'trajectory.json').read_text())
                rec['attempts']=meta.get('attempted_updates',len(recorded_rows))
                rec['gradient_fallback_count']=sum(bool(r.get('gradient_fallback')) for r in recorded_rows)
                rec['accepted_gradient_fallback_count']=sum(bool(r.get('gradient_fallback')) and bool(r.get('accepted')) for r in recorded_rows)
            except (OSError, ValueError) as exc:
                rec['status'] = f'MISSING/INVALID final.json: {exc}'
            try:
                with np.load(folder/'final.npz', allow_pickle=False) as data:
                    saved_points = data['points'] if 'points' in data else points
                    if saved_points is None:
                        raise ValueError('points absent in both input and final')
                    if points is not None and not np.array_equal(saved_points, points):
                        raise ValueError('final points differ from target input points')
                    cube, axes, edges = grid(saved_points, data['chi'])
                rec.update(cube=cube, axes=axes, edges=edges)
            except (OSError, ValueError, KeyError) as exc:
                rec['status'] += f'\nNO VALID FINAL ARRAY: {exc}'
            scene[policy] = rec
            receipt['records'].append(dict(target=name, policy=policy, status=rec['status'],
                                           plotted_final_array='cube' in rec))
        scenes[name] = scene
    return scenes, receipt


def compact_status(rec):
    status=rec.get('status','STATUS_UNKNOWN').split('\n')[0]
    return '' if status in ('MAX_UPDATES','budget-capped [MAX_UPDATES]') else status


def title(ax, name, method, rec):
    if method == 'truth':
        text=LABELS[method]+'\n'+TARGETS[name]['label']
    else:
        value=rec.get('metrics',{}).get('complex_nrmse')
        nrmse=f'{value:.3f}' if isinstance(value,(int,float)) else 'unavailable'
        updates=rec.get('updates')
        fallback=rec.get('gradient_fallback_count','?')
        attempts=rec.get('attempts','?')
        text=LABELS[method]+f'\nNRMSE={nrmse}\nG={fallback}/{attempts} | n={updates if updates is not None else "?"}'
        status=compact_status(rec)
        if status:text+='\n'+status
    ax.set_title(text, fontsize=8)


def surface_plot(scenes, out, names=None):
    names = tuple(TARGETS) if names is None else tuple(names)
    nrows = len(names)
    fig = plt.figure(figsize=(23, 3.7*nrows+1.5), layout='constrained')
    # Compute from the complete scene dictionary, including when plotting a
    # single target: every overview/row shares exactly the same global scale.
    real_arrays=[r['cube'].real for scene in scenes.values() for r in scene.values() if 'cube' in r]
    global_vmax=max([1.5]+[float(values.max()) for values in real_arrays])
    global_vmin=min([0.]+[float(values.min()) for values in real_arrays])
    norm = Normalize(global_vmin, global_vmax, clip=False)
    cmap = plt.get_cmap('viridis')
    for row, name in enumerate(names):
        for col, method in enumerate(('truth',)+POLICIES):
            ax = fig.add_subplot(nrows, 7, row*7+col+1, projection='3d')
            rec = scenes[name][method]
            title(ax, name, method, rec)
            if 'cube' in rec:
                cube = rec['cube']
                coords = np.meshgrid(*rec['edges'], indexing='ij')
                ax.voxels(*coords, cube.real >= THRESHOLD, facecolors=cmap(norm(cube.real)),
                          edgecolor=(.1, .1, .1, .18), linewidth=.10, shade=False)
                if not np.any(cube.real >= THRESHOLD):
                    ax.text2D(.04, .04, 'EMPTY support at Re chi >= .5', transform=ax.transAxes, fontsize=7)
            else:
                ax.text2D(.03, .35, rec['status'][:150], transform=ax.transAxes, color='firebrick', fontsize=7, wrap=True)
            ax.set(xlim=(-.5,.5), ylim=(-.5,.5), zlim=(-.5,.5))
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=50, azim=-56)
            ax.set_xticks([-.5,0,.5]); ax.set_yticks([-.5,0,.5]); ax.set_zticks([-.5,0,.5])
            ax.tick_params(labelsize=5)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=fig.axes, shrink=.5, label=f'Re chi (global saved range [{global_vmin:.4g}, {global_vmax:.4g}]); shown geometry: Re chi >= .5')
    fig.suptitle('Final reconstructions | shown geometry: Re chi >= 0.5\n'+FIGURE_KEY, fontsize=13)
    filename='FINAL_SUPPORT_6x7.png' if nrows==6 else f'FINAL_SUPPORT_{names[0]}_1x7.png'
    fig.savefig(out/filename, dpi=150)
    plt.close(fig)


def slice_plots(scenes, out, mode='central'):
    # Three physical orientations per component; central or every native layer.
    all_cubes = [r['cube'] for s in scenes.values() for r in s.values() if 'cube' in r]
    for component, cmap in (('real','viridis'), ('imag','magma')):
        arrays = [getattr(c, component) for c in all_cubes]
        lo = min([0.] + [float(a.min()) for a in arrays])
        hi = max([1.5 if component == 'real' else .15] + [float(a.max()) for a in arrays])
        norm = Normalize(lo, hi)
        for axis, plane in enumerate(('YZ', 'XZ', 'XY')):
            # Central slices use the native layer nearest zero, never a truth-selected slice.
            nmax = max([c.shape[axis] for c in all_cubes], default=12)
            for layer in ([None] if mode == 'central' else range(nmax)):
                fig, axes = plt.subplots(6,7, figsize=(21,18), layout='constrained')
                for row, name in enumerate(TARGETS):
                    for col, method in enumerate(('truth',)+POLICIES):
                        ax = axes[row,col]; rec = scenes[name][method]; title(ax,name,method,rec)
                        native_layer=int(np.argmin(np.abs(rec['axes'][axis]))) if layer is None and 'cube' in rec else layer
                        if 'cube' in rec and native_layer < rec['cube'].shape[axis]:
                            vals = getattr(rec['cube'], component)
                            image = np.take(vals,native_layer,axis=axis)
                            ax.imshow(image.T, origin='lower', extent=(-.5,.5,-.5,.5),
                                      cmap=cmap, norm=norm, interpolation='nearest')
                            coord = rec['axes'][axis][native_layer]
                            ax.text(.01,.01,f'{"xyz"[axis]}={coord:.5f}',transform=ax.transAxes,fontsize=6,color='white',bbox={'facecolor':'black','alpha':.5,'pad':1})
                        else:
                            ax.text(.03,.4,rec['status'][:100],transform=ax.transAxes,fontsize=6,color='firebrick',wrap=True)
                        transverse=[i for i in range(3) if i!=axis]
                        ax.set_xlabel('xyz'[transverse[0]],fontsize=6);ax.set_ylabel('xyz'[transverse[1]],fontsize=6)
                        ax.set_xticks([-.5,0,.5]);ax.set_yticks([-.5,0,.5]);ax.tick_params(labelsize=5)
                fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axes.ravel().tolist(),shrink=.5,label=f'{component.title()} chi | common all-target/all-method/all-layer scale [{lo:.4g},{hi:.4g}]')
                slice_label='central native layer nearest zero' if layer is None else f'layer {layer}'
                suffix='central' if layer is None else f'layer{layer:02d}'
                fig.suptitle(f'Raw final {component} chi | {plane} | {slice_label}\n'+FIGURE_KEY,fontsize=12)
                fig.savefig(out/f'RAW_{component.upper()}_{plane}_{suffix}.png',dpi=120)
                plt.close(fig)


def heatmap(scenes, out):
    values = np.full((6,6),np.nan)
    annotations = np.empty((6,6),object)
    for row,name in enumerate(TARGETS):
        for col,policy in enumerate(POLICIES):
            rec=scenes[name][policy];value=rec.get('metrics',{}).get('complex_nrmse')
            if 'cube' in rec and isinstance(value,(int,float)) and np.isfinite(value):
                values[row,col]=value
                annotations[row,col]=f'{value:.3f}\nG={rec.get("gradient_fallback_count","?")}/{rec.get("attempts","?")} | n={rec.get("updates","?")}'
                status=compact_status(rec)
                if status:annotations[row,col]+='\n'+status
            else:
                annotations[row,col]=rec['status'].split('\n')[0]+'\nmetric/array unavailable'
    fig,ax=plt.subplots(figsize=(17,9),layout='constrained')
    cmap=plt.get_cmap('magma').copy();cmap.set_bad('#dedede')
    vmax=max(float(np.nanmax(values)),1e-8) if np.isfinite(values).any() else 1.
    im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,vmin=0,vmax=vmax,aspect='auto')
    for (row,col),txt in np.ndenumerate(annotations):
        ax.text(col,row,txt,ha='center',va='center',fontsize=7,color='white' if np.isfinite(values[row,col]) and values[row,col]>.45*vmax else 'black',wrap=True)
    ax.set_xticks(range(6),[LABELS[p] for p in POLICIES],rotation=18,ha='right')
    ax.set_yticks(range(6),[TARGETS[n]['label'] for n in TARGETS])
    ax.set_title('Final complex NRMSE | lower is better\n'+FIGURE_KEY)
    fig.colorbar(im,ax=ax,label='Final complex NRMSE (saved final.json)')
    fig.savefig(out/'FINAL_COMPLEX_NRMSE_6x6.png',dpi=160);plt.close(fig)



def trajectory_plots(root, scenes, out):
    """Saved material NRMSE against accepted count and recorded cumulative time.

    A row stores wall_s, not a total wall timestamp. The time curve therefore
    uses model setup + sum(recorded row wall), explicitly labeled as such.
    Saved final.wall_s is a distinct square marker for actual total wall.
    """
    colors=plt.get_cmap('tab10').colors
    for name in TARGETS:
        fig, axes=plt.subplots(1,2,figsize=(15,5.5),layout='constrained')
        notices=[]
        for idx, policy in enumerate(POLICIES):
            folder=root/'results'/name/policy
            try:rows=json.loads((folder/'trajectory.json').read_text())
            except (OSError,ValueError):rows=[]
            try:final=json.loads((folder/'final.json').read_text())
            except (OSError,ValueError):final={}
            initial=(rows[0].get('before_metrics') or {}).get('complex_nrmse') if rows else None
            setup=float(final.get('model_setup_s') or 0.)
            x=[];times=[];y=[];accepted_count=0;elapsed=setup
            if initial is not None:x.append(0);times.append(setup);y.append(initial)
            for row in rows:
                elapsed+=float(row.get('wall_s') or 0.)
                accepted_count+=int(bool(row.get('accepted')))
                value=(row.get('metrics') or {}).get('complex_nrmse')
                if isinstance(value,(int,float)) and np.isfinite(value):
                    x.append(row.get('accepted_update_count',accepted_count));times.append(elapsed);y.append(value)
            status=status_label(final.get('status','RUNNING_OR_NOT_RUN'))
            fallback_count=sum(bool(r.get('gradient_fallback')) for r in rows)
            accepted_fallback=sum(bool(r.get('gradient_fallback')) and bool(r.get('accepted')) for r in rows)
            label=LABELS[policy]+f' | G={fallback_count}/{len(rows)} | n={final.get("updates","?")}'
            panel_stop=compact_status({'status':status})
            if panel_stop:label+=' | '+panel_stop
            if rows and y:
                axes[0].plot(x,y,'-o',ms=3,lw=1.3,color=colors[idx],label=label)
                axes[1].plot(times,y,'-o',ms=3,lw=1.3,color=colors[idx],label=label)
            else:
                notices.append(policy+': '+status+'; no saved trajectory')
            last=(final.get('metrics') or {}).get('complex_nrmse')
            if isinstance(last,(int,float)) and np.isfinite(last):
                axes[0].scatter([final.get('updates',accepted_count)],[last],marker='s',s=38,color=colors[idx],zorder=4)
                if isinstance(final.get('wall_s'),(int,float)):
                    axes[1].scatter([final['wall_s']],[last],marker='s',s=38,color=colors[idx],zorder=4)
        axes[0].set_xlabel('Completed accepted material updates')
        axes[1].set_xlabel('Cumulative setup + recorded update wall (s); square = total final wall')
        for ax in axes:
            ax.set_ylabel('Complex material NRMSE');ax.grid(alpha=.25)
            handles,labels=ax.get_legend_handles_labels()
            if handles:ax.legend(fontsize=6.7)
        if notices:axes[1].text(.02,.98,'\n'.join(notices),transform=axes[1].transAxes,va='top',fontsize=7,color='firebrick')
        fig.suptitle(TARGETS[name]['label']+' | final material trajectories\n'+FIGURE_KEY,fontsize=11)
        fig.savefig(out/f'TRAJECTORY_{name}.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--slice-mode',choices=('central','all'),default='central',help='central exports 6 native-slice figures; all exports every layer')
    args=parser.parse_args();root=args.root.resolve()
    scenes,receipt=read(root);out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    surface_plot(scenes,out);heatmap(scenes,out)
    for name in TARGETS:surface_plot(scenes,out,names=(name,))
    slice_plots(scenes,out,mode=args.slice_mode)
    trajectory_plots(root,scenes,out)
    receipt['slice_status']='CENTRAL_NATIVE_LAYERS_ALL_3_PLANES' if args.slice_mode=='central' else 'ALL_RAW_LAYERS_ALL_3_PLANES'
    receipt['camera']={'elev':50,'azim':-56}
    real_arrays=[r['cube'].real for scene in scenes.values() for r in scene.values() if 'cube' in r]
    receipt['surface_color_limits']=[min([0.]+[float(v.min()) for v in real_arrays]),max([1.5]+[float(v.max()) for v in real_arrays])]
    receipt['surface_geometry_scope']='Only support Re chi >= .5 is rendered; background is hidden. Colors share the complete saved-array global real range.'
    receipt['outputs']=[p.name for p in sorted(out.glob('*.png'))]
    (out/'PLOTTING_RECEIPT.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
