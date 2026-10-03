"""Explicit baseline pinning; closure has a separate observed source epoch."""
from pathlib import Path
import hashlib
import importlib
import json
import os
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = None
BASELINE_LOCK = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(baseline_root, baseline_lock=None):
    global BASELINE, BASELINE_LOCK
    baseline = Path(baseline_root).expanduser().resolve()
    lock_path=Path(baseline_lock).resolve() if baseline_lock else ROOT/'research/A17_BASELINE_LOCK.json'
    lock=json.loads(lock_path.read_text())
    if lock.get('prior_paid_occupation_s')!=4889.515 or lock.get('cumulative_limit_s')!=43200:
        raise ValueError('baseline occupation lock differs from authorized budget')
    expected_names={p.relative_to(baseline).as_posix() for p in (baseline/'code').glob('*.py')}|{'A17_RUN_CONFIG.json','SOURCE_MANIFEST.json','PROTOCOL.md'}
    if set(lock['read_only_files'])!=expected_names:
        raise ValueError('baseline lock source names differ from declared source epoch')
    for path,expected in lock['read_only_files'].items():
        if sha(baseline/path)!=expected:raise ValueError('locked baseline drift: '+path)
    manifest = json.loads((baseline / 'SOURCE_MANIFEST.json').read_text())
    if len(manifest['modules']) != 42:
        raise ValueError('expected unchanged 42-module baseline manifest')
    for name, row in manifest['modules'].items():
        path = baseline / row['path']
        if sha(path) != row['sha256']:
            raise ValueError('baseline source drift: ' + name)
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve() != path.resolve():
            raise RuntimeError('already imported a different vendor: ' + name)
    loaded = sys.modules.get('pinned_runtime')
    if loaded is not None and Path(loaded.ROOT).resolve() != baseline:
        raise RuntimeError('already imported a different baseline')
    if BASELINE is not None and BASELINE != baseline:
        raise RuntimeError('one baseline per fresh process')
    sys.path.insert(0, str(baseline / 'code'))
    runtime = importlib.import_module('pinned_runtime')
    if Path(runtime.ROOT).resolve() != baseline:
        raise RuntimeError('explicit baseline root mismatch')
    BASELINE = baseline
    BASELINE_LOCK=lock_path
    return runtime


def check_config_sources(config, config_path):
    """Pre-result lock validation. No measured outcome is consulted."""
    if config.get('baseline_lock_sha256') != sha(BASELINE_LOCK):
        raise ValueError('config must bind unchanged baseline lock')
    sources=config.get('deployment_sources',config.get('closure_source_hashes',config.get('source_hashes')))
    if not sources:
        if config.get('schema')!='a17.closure.prelock.replay.v1' or config.get('permitted_modes')!=['replay'] or config.get('lock_status')!='PRELOCK_REPLAY_ONLY':
            raise ValueError('final config requires deployment_sources; prelock permits replay only')
        sources={}
    if isinstance(sources,list):sources={row['path']:row['sha256'] for row in sources}
    sources={path:(value if isinstance(value,str) else value['sha256']) for path,value in sources.items()}
    required={'code/'+name for name in ['closure_runtime.py','closure_state.py','verified_exchange.py','run_closure.py','online_tolerance.py']}
    if sources and not required.issubset(sources):raise ValueError('missing locked closure source')
    for path,expected in sources.items():
        local=ROOT/path
        if not local.resolve().is_relative_to(ROOT) or sha(local)!=expected:
            raise ValueError('locked closure source drift: '+path)
    if config.get('online_tolerance_sha256')!=sha(ROOT/'code/online_tolerance.py'):
        raise ValueError('locked acceptance source drift')
    return dict(path=str(Path(config_path).resolve()),sha256=sha(config_path),source_hashes=sources,
                baseline_lock_sha256=sha(BASELINE_LOCK))


def receipt():
    if BASELINE is None:
        raise RuntimeError('initialize explicit baseline first')
    import pinned_runtime
    actual = {}
    roots = (BASELINE, ROOT)
    for name, module in list(sys.modules.items()):
        source = getattr(module, '__file__', None)
        if source is None:
            continue
        path = Path(source).resolve()
        if path.is_file() and any(path.is_relative_to(root) for root in roots):
            actual[name] = dict(actual_path=str(path), sha256=sha(path))
    return dict(schema='a17.closure.runtime.v1', baseline_root=str(BASELINE),
                baseline_manifest_sha256=sha(BASELINE / 'SOURCE_MANIFEST.json'),
                baseline_lock_sha256=sha(BASELINE_LOCK),
                baseline=pinned_runtime.receipt(), actual_imports=actual,
                closure_sources={str(p.relative_to(ROOT)): sha(p)
                                 for p in sorted((ROOT / 'code').glob('*.py'))},
                python=platform.python_version(), executable=sys.executable,
                threads={k: os.getenv(k) for k in
                         ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                numerical_monkeypatch=False)
