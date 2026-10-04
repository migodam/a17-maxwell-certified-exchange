"""Hash-bound new frozen inputs and unchanged physical material coordinates."""
from pathlib import Path
import json
import time
import numpy as np
from scipy import linalg as la
from closure_runtime import sha


def manifest_map(manifest):
    rows = manifest.get('inputs', manifest.get('files', manifest.get('artifacts')))
    if rows is None:
        raise ValueError('input manifest needs inputs/files/artifacts')
    if isinstance(rows, dict):
        return {str(k).replace('\\', '/'): v if isinstance(v, str) else v['sha256']
                for k, v in rows.items()}
    result = {}
    for row in rows:
        path = str(row['path']).replace('\\', '/')
        if path in result and result[path] != row['sha256']:
            raise ValueError('conflicting manifest input hash')
        result[path] = row['sha256']
    return result


def check_input(path, inputs_root, manifest_path, known):
    path = Path(path).resolve()
    roots = [Path(inputs_root).resolve(), Path(manifest_path).resolve().parent]
    keys = [path.relative_to(root).as_posix() for root in roots if path.is_relative_to(root)]
    keys += ['inputs/' + path.relative_to(Path(inputs_root).resolve()).as_posix()]
    matches = [known[k] for k in keys if k in known]
    actual = sha(path)
    if not matches or any(value != actual for value in matches):
        raise ValueError('missing/drifted declared input: ' + path.name)
    return dict(path=str(path), sha256=actual)


def load_state(inputs_root, object_id, phase, device, manifest_path,
               phases=None, noise_basis_points=0, realization_index=0):
    from a10_common import DenseDDA, Tangent
    from scenes import acquisition
    from material_gauge_persistence import load_frozen_material
    from operators import make_problem
    from maxwell_state import array_hash
    started = time.perf_counter()
    inputs_root, manifest_path = Path(inputs_root).resolve(), Path(manifest_path).resolve()
    known = manifest_map(json.loads(manifest_path.read_text()))
    phases = phases or dict(early=0, middle=3, late=17)
    it = phases[phase]
    ref = inputs_root / f'object_{object_id}'
    paths = [inputs_root / 'scenes.json', ref / 'common_data.npz',
             ref / f'anchor_{it:02d}.npz', inputs_root / 'original_resolved_config.json']
    receipts = [check_input(p, inputs_root, manifest_path, known) for p in paths]
    scene_rows = json.loads(paths[0].read_text())
    scene_rows = scene_rows['scenes'] if isinstance(scene_rows, dict) else scene_rows
    scenes = [s for s in scene_rows if s['object_id'] == object_id]
    if len(scenes) != 1:
        raise ValueError('object must appear exactly once in declared scenes')
    scene = scenes[0]
    common, anchor = [dict(np.load(p, allow_pickle=False)) for p in paths[1:3]]
    original_config=json.loads(paths[3].read_text())
    points = common['points']; volume = float(scene['edge'] / scene['n']) ** 3
    tangent, gauge, ell = load_frozen_material(points, volume, scene['representation'],
                                               common, anchor, tangent_class=Tangent,
                                               allow_legacy=False)
    dirs, pols, rx, obs = acquisition(scene)
    model = DenseDDA(points, volume, scene['wavenumber'], dirs, pols, rx, obs, device=device)
    state = model.state(anchor['chi'])
    noise = measurement_noise(common['data0'], float(common['scale']), object_id,
                              noise_basis_points, realization_index)
    data = common['data0'] + noise['values']
    ctx, ev = make_problem(state, tangent, data, float(common['scale']),
                           float(anchor['lambda_total']), ell)
    ctx.material_gauge = gauge
    return dict(scene=scene, common=common, anchor=anchor, state=state,
                ctx=ctx, ev=ev, tangent=tangent, original_config=original_config,
                physical_setup_wall_s=time.perf_counter() - started,
                inputs=receipts, input_manifest_sha256=sha(manifest_path),
                state_hash=array_hash(anchor['chi']), phase=phase, iteration=it,
                noise=noise, evaluation_reference_locator=dict(inputs_root=str(inputs_root),
                    manifest_path=str(manifest_path),path=str(ref/f'step_{it:02d}.npz')))


def load_evaluation_reference(bundle):
    """Offline only: called after every online k endpoint is durably saved."""
    from material_gauge_persistence import load_warm_step
    locator=bundle['evaluation_reference_locator']
    if sha(locator['manifest_path'])!=bundle['input_manifest_sha256']:
        raise ValueError('input manifest changed after online preparation')
    manifest=json.loads(Path(locator['manifest_path']).read_text())
    rec=check_input(locator['path'],locator['inputs_root'],locator['manifest_path'],manifest_map(manifest))
    saved=dict(np.load(locator['path'],allow_pickle=False))
    step=load_warm_step(bundle['tangent'],saved,bundle['ctx'].material_gauge,allow_legacy=False)
    bundle.update(saved=saved,full_step=step,original_saved_full_step=step.copy())
    return rec


def measurement_noise(data, scale, object_id, basis_points, realization_index):
    from maxwell_state import array_hash
    if basis_points not in (0, 100, 300) or realization_index not in (0, 1, 2):
        raise ValueError('noise supports locked 0/1/3 percent and realization0/1/2')
    data = np.asarray(data)
    if data.ndim != 2 or not np.iscomplexobj(data) or not np.isfinite(data).all():
        raise ValueError('complex (illumination, observation) data required')
    data_norm = float(la.norm(data)); relative_norm = basis_points / 10000.
    if scale <= 0 or not np.isclose(scale, data_norm, rtol=1e-12, atol=0.):
        raise ValueError('measurement scale must equal original data L2 norm')
    seed = [20261002, int(object_id), int(basis_points), int(realization_index)]
    noise = np.zeros_like(data, dtype=np.complex128)
    if basis_points:
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        noise = rng.normal(size=data.shape) + 1j*rng.normal(size=data.shape)
        noise *= relative_norm * data_norm / la.norm(noise)
    return dict(schema='a17.fixed.complex.measurement_noise.v1', seed_sequence=seed,
                relative_norm=relative_norm, basis_points=basis_points,
                realization_index=realization_index, noise_hash=array_hash(noise),
                original_data_hash=array_hash(data), noisy_data_hash=array_hash(data+noise),
                data_norm=data_norm, noise_norm=float(la.norm(noise)),
                normalized_noise_norm=float(la.norm(noise)/scale),
                scope='Independent complex Gaussian measurement noise; fixed norm eta*norm(data0); residual uses prediction-data0-noise',
                values=noise)


def persisted_workspace(bundle, persisted):
    """Recreate precisely the paid public coefficient maps; no new candidate bank."""
    from operators import SelectorContext, ReducedModel
    from selectors_current import CandidatePool
    from pairtsom_adapter import PhaseLedger
    from types import SimpleNamespace
    from maxwell_state import array_hash
    persisted = Path(persisted).resolve()
    path = persisted / 'public_workspace.npz'
    with np.load(path, allow_pickle=False) as handle:
        bank = {key: handle[key].copy() for key in handle.files}
    receipt_path = persisted / 'runtime_receipt.json'
    rec = json.loads(receipt_path.read_text())
    ctx = bundle['ctx']
    if rec['state_hash'] != bundle['state_hash'] or rec['material_gauge']['basis_fingerprint'] != ctx.material_gauge['basis_fingerprint']:
        raise ValueError('persisted state/material gauge mismatch')
    if rec['pool_hash'] != array_hash(bank['pool']):
        raise ValueError('persisted pool hash mismatch')
    if float(bank['lambda_total']) != ctx.lam or bank['PB'].shape[0] != ctx.P:
        raise ValueError('persisted prior/illumination mismatch')
    for key, current in [('r', ctx.r), ('ell', ctx.ell)]:
        if bank[key].shape != current.shape or not np.array_equal(bank[key], current):
            raise ValueError('persisted public ' + key + ' differs from physical state')
    c = SelectorContext(); c.P, c.n_current, c.q = bank['PB'].shape
    c.m = bank['S'].shape[0]; c.dim_material = 2*c.q
    c.r, c.ell, c.lam = bank['r'], bank['ell'], float(bank['lambda_total'])
    c.ledger = PhaseLedger(); c.material_gauge = ctx.material_gauge
    c._L = lambda U: bank['L'] @ U
    c._S = lambda U: bank['S'] @ U
    c._B = lambda U: np.stack([U.conj().T @ p for p in bank['PB']])
    anchor = ReducedModel(c, bank['anchor'])
    pool = CandidatePool(bank['pool'], list(bank['labels']), list(bank['families']))
    w = SimpleNamespace(ctx=c, pool=pool, anchor=anchor, Q=bank['Q'], metadata={})
    return dict(workspace=w, L=bank['L'], S=bank['S'], PB=bank['PB'],
                pool_hash=array_hash(pool.vectors), persisted_hashes={str(path): sha(path),
                str(receipt_path): sha(receipt_path)}, workspace_wall_s=None,
                cache_scope='Persisted public cache, not independent cold workspace acquisition')
