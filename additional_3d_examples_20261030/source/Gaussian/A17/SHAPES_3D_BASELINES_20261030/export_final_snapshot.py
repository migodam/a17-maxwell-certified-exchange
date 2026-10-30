"""Copy stopped-run evidence to a transfer ZIP, without importing the study."""
from pathlib import Path
import json
import zipfile

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT.parents[3]/'RETURN_FINAL_20261030.zip'
entries=[]
with zipfile.ZipFile(OUTPUT,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
    for folder in ('inputs','results','gradient_control','execution','verification'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and path.suffix!='.zip' and '__pycache__' not in path.parts:
                relative=path.relative_to(ROOT).as_posix()
                archive.write(path,relative)
                entries.append(dict(path=relative,bytes=path.stat().st_size))
    for pattern in ('*.json','*.jsonl'):
        for path in sorted(ROOT.glob(pattern)):
            archive.write(path,path.name)
            entries.append(dict(path=path.name,bytes=path.stat().st_size))
    for name in ('run_i9.log','run_gpu.log','run_parallel.log','run_fast.log',
                 'run_extended.log','run_serial_resume.log','run_namespace_resume.log',
                 'run_gradient_serial.log'):
        path=ROOT.parents[3]/name
        if path.exists():
            archive.write(path,'execution/remote_logs/'+name)
            entries.append(dict(path='execution/remote_logs/'+name,bytes=path.stat().st_size))
    archive.writestr('execution/RETURN_FILE_LIST.json',json.dumps(entries,indent=2))
print(json.dumps(dict(archive=str(OUTPUT),files=len(entries),bytes=OUTPUT.stat().st_size)))
