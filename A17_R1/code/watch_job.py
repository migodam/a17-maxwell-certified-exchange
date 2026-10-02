"""Bounded local-to-remote A17 job watchdog, not a persistent scheduler."""
from pathlib import Path
import argparse,os,time,json,subprocess,sys,uuid,datetime

ROOT=Path(__file__).resolve().parents[1]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def log(p,x):
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def main():
    import psutil
    p=argparse.ArgumentParser();p.add_argument('--unit',required=True);p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
    command=args.command[1:] if args.command and args.command[0]=='--' else args.command
    ex=ROOT/'execution';ex.mkdir(exist_ok=True);ledger=ex/'gpu_attempts.jsonl';lock=ex/'GPU_JOB_LOCK.json'
    records=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    starts={x['attempt'] for x in records if x['event']=='start'};ends={x['attempt'] for x in records if x['event']=='end'}
    if starts-ends or lock.exists():raise RuntimeError('Unresolved existing A17 attempt/lock, no launch')
    charged=sum(x['occupation_s'] for x in records if x['event']=='end')
    if charged>=43200:raise RuntimeError('A17 12h budget exhausted')
    # Do not overlap another project numerical Python job; read-only preflight.
    outsiders=[]
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.pid==os.getpid():continue
        s=' '.join(process.info.get('cmdline') or [])
        if 'python' in (process.info.get('name') or '').lower() and any(x in s for x in ('run_validation_queue.py','run_continuation.py','optimized_child.py','run_exchange.py','verify_cuda.py')):outsiders.append(dict(pid=process.pid,command=s))
    if outsiders:raise RuntimeError('Active numerical processes: '+json.dumps(outsiders))
    smi=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    if smi.returncode:raise RuntimeError('nvidia-smi preflight failure')
    total,used,util=[float(x.strip()) for x in smi.stdout.splitlines()[0].split(',')]
    if used/total>=.8:raise RuntimeError('Preflight GPU memory >=80%')
    attempt='a17_'+args.unit+'_'+uuid.uuid4().hex[:10]
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,json.dumps(dict(attempt=attempt,watchdog_pid=os.getpid(),created=now())).encode());os.close(fd)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    wall=time.monotonic();limit=min(7200,43200-charged);child=None;stop=None;exitcode=None;peak=used
    logfile=(ex/(attempt+'.log')).open('w',encoding='utf-8')
    telemetry=ex/(attempt+'_resources.jsonl')
    try:
        child=subprocess.Popen(command,stdout=logfile,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
        log(ledger,dict(event='start',attempt=attempt,unit=args.unit,utc=now(),watchdog_pid=os.getpid(),child_pid=child.pid,command=command,charged_before_s=charged,limit_s=limit,cpu_threads=4,preflight=dict(total_mib=total,used_mib=used,gpu_util=util),scope='Complete job occupied interval including CPU/setup/teacher/offline/failures; not CUDA active time'))
        print(json.dumps(dict(event='started',attempt=attempt,pid=child.pid,unit=args.unit)),flush=True)
        missed=0;warned=False
        while child.poll() is None:
            elapsed=time.monotonic()-wall
            if elapsed>=limit:stop='TIME_OR_TOTAL_BUDGET_LIMIT';break
            try:
                q=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,utilization.memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                if q.returncode:raise RuntimeError(q.stderr)
                mem,gu,mu=[float(x.strip()) for x in q.stdout.splitlines()[0].split(',')];peak=max(peak,mem);missed=0
                process=psutil.Process(child.pid)
                log(telemetry,dict(elapsed_s=elapsed,memory_mib=mem,gpu_util=gu,memory_util=mu,host_rss=process.memory_info().rss,host_cpu_times=list(process.cpu_times()[:2])))
                if mem/total>=.9:stop='GPU_MEMORY_90PCT_STOP';break
                if mem/total>=.8 and not warned:print('GPU_MEMORY_80PCT_WARNING',flush=True);warned=True
            except psutil.NoSuchProcess:break
            except Exception as exc:
                missed+=1;log(telemetry,dict(elapsed_s=elapsed,error=str(exc),consecutive=missed))
                if missed>=3:stop='MONITOR_FAILURE';break
            time.sleep(1)
        if stop and child.poll() is None:
            # Only this new A17 child and its descendants; preserve saved safe cells.
            proc=psutil.Process(child.pid);children=proc.children(recursive=True)
            for c in children:c.terminate()
            proc.terminate();_,alive=psutil.wait_procs([proc]+children,timeout=10)
            for c in alive:c.kill()
            psutil.wait_procs(alive,timeout=10)
        exitcode=child.wait(timeout=15)
    finally:
        logfile.close();elapsed=time.monotonic()-wall
        if child is not None and child.poll() is None:
            log(ledger,dict(event='monitor_error',attempt=attempt,unit=args.unit,utc=now(),error='Child status unknown; lock retained'))
            raise RuntimeError('A17 child still active/unknown, lock retained')
        log(ledger,dict(event='end',attempt=attempt,unit=args.unit,utc=now(),exit_code=exitcode,occupation_s=elapsed,stop_reason=stop,peak_device_used_mib=peak,remaining_s=43200-charged-elapsed))
        lock.unlink()
        print(json.dumps(dict(event='ended',attempt=attempt,occupation_s=elapsed,exit_code=exitcode,stop_reason=stop)),flush=True)
    if stop or exitcode:raise SystemExit(1)
if __name__=='__main__':main()
