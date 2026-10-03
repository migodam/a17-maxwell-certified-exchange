"""Serial saved-state jobs only; the outer resource guard owns this process."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, time

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline-root',required=True);p.add_argument('--config',required=True)
    p.add_argument('--mode',choices=['replay','validation','noise'],required=True)
    p.add_argument('--object',type=int,required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out)
    if out.exists():raise ValueError('new batch directory required')
    if a.mode not in cfg['permitted_modes']:raise ValueError('mode outside source lock')
    key={'replay':'replay_objects','validation':'validation_objects','noise':'noise_objects'}[a.mode]
    if a.object not in cfg[key]:raise ValueError('object outside source lock')
    if a.mode=='noise':
        tasks=[('middle',b,j) for b in (100,300) for j in range(3)]
    else:tasks=[(s,0,0) for s in ('early','middle','late')]
    out.mkdir(parents=True);rows=[];started=time.perf_counter()
    for phase,basis_points,rep in tasks:
        folder=out/(phase if not basis_points else f'{phase}_noise{basis_points}_rep{rep}')
        cmd=[sys.executable,'-B',str(root/'code/run_closure.py'),'--baseline-root',a.baseline_root,
             '--baseline-lock',str(root/cfg['baseline_lock_path']),
             '--config',a.config,'--mode',a.mode,'--object',str(a.object),'--phase',phase,
             '--device','cuda','--out',str(folder)]
        if a.mode!='replay':cmd+=['--inputs-root',str(root/'inputs'),'--input-manifest',str(root/'inputs/INPUT_MANIFEST.json')]
        if basis_points:cmd+=['--noise-basis-points',str(basis_points),'--realization-index',str(rep)]
        t=time.perf_counter();rc=subprocess.call(cmd)
        status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {'status':'UNKNOWN'}
        row=dict(phase=phase,noise_basis_points=basis_points,realization=rep,returncode=rc,
                 status=status['status'],wall_s=time.perf_counter()-t,path=str(folder))
        rows.append(row)
        (out/'batch_receipt.json').write_text(json.dumps(dict(mode=a.mode,object=a.object,
            config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
            tasks=rows,total_wall_s=time.perf_counter()-started),indent=2)+'\n')
        if rc or status['status']!='COMPLETED':raise SystemExit('State failed; no retry or additional state launched')
    print(json.dumps(dict(event='batch_completed',mode=a.mode,object=a.object,states=len(rows))),flush=True)

if __name__=='__main__':main()
