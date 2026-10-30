"""Path-only importer for ORIGINAL A17 numerical kernels and isolated wiring.

Original kernels are unmodified; isolated controller/helper copies preserve their
numerical definitions. No old runner,
source manifest, closed budget, frozen input, or SHA check is loaded. Call in a
fresh interpreter. Import-only verification is safe; constructing DenseDDA or
calling any state/action spends a new experiment budget and belongs to caller.

verified_exchange imports safely but its historical lazy routes remain blocked.
Use original_exchange_controller.run_original_exchange for the isolated wiring
which redirects only those routes and replaces logging hashes with descriptors.
"""
from pathlib import Path
import importlib
import importlib.abc
import importlib.util
import sys

PROJECT = Path(__file__).resolve().parents[6]
A17 = PROJECT / "Gaussian/A17"
VENDOR = A17 / "EXCHANGE_R1/vendor/Gaussian"
SOURCES = {
    "a9_engine": VENDOR / "A9/code/a9_engine.py",
    "experiment_support": VENDOR / "A9/code/experiment_support.py",
    "a10_common": VENDOR / "A10/code/a10_common.py",
    "anchored_rom": VENDOR / "A10/code/anchored_rom.py",
    "cair": VENDOR / "A10/code/cair.py",
    "feasible_step": VENDOR / "A10/code/feasible_step.py",
    "reduced_normal": VENDOR / "A10/code/reduced_normal.py",
    "campaign_common": VENDOR / "A12/code/campaign_common.py",
    "galerkin_rom": VENDOR / "A12/code/galerkin_rom.py",
    "pairtsom_adapter": VENDOR / "A14/src/pairtsom_adapter.py",
    "fast_contractions": VENDOR / "A16/CURRENT_SELECTION_R1/code/fast_contractions.py",
    "operators": VENDOR / "A16/CURRENT_SELECTION_R1/code/operators.py",
    "selectors_current": VENDOR / "A16/CURRENT_SELECTION_R1/code/selectors_current.py",
    "selector_coordinates": VENDOR / "A16/CURRENT_SELECTION_R1/code/selector_coordinates.py",
    "scenes": VENDOR / "A16/CURRENT_SELECTION_R1/code/scenes.py",
    "endpoint_batch": A17 / "EXCHANGE_R1/code/endpoint_batch.py",
    "removal_core_batch": A17 / "EXCHANGE_R1/code/removal_core_batch.py",
    "online_tolerance": A17 / "CLOSURE_R1/code/online_tolerance.py",
    "verified_exchange": A17 / "CLOSURE_R1/code/verified_exchange.py",
    "original_exchange_helpers": Path(__file__).resolve().parent / "original_exchange_helpers.py",
    "original_exchange_controller": Path(__file__).resolve().parent / "original_exchange_controller.py",
}
BLOCKED = frozenset({"pinned_runtime", "closure_runtime", "closure_state", "closure_state_v2",
                     "run_exchange", "maxwell_state", "run_closure", "run_closure_v2",
                     "run_nonlinear", "run_locked_batch", "run_locked_batch_v2", "run_locked_batch_v3"})

class OriginalKernelFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in BLOCKED:
            raise ImportError(f"Historical A17 runner forbidden in this isolated adapter: {fullname}")
        source = SOURCES.get(fullname)
        if source is None:
            return None
        return importlib.util.spec_from_file_location(fullname, source)

_FINDER = None

def install():
    global _FINDER
    for name in BLOCKED:
        if name in sys.modules:
            raise RuntimeError(f"Fresh interpreter required; historical runner already loaded: {name}")
    for name, source in SOURCES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve() != source.resolve():
            raise RuntimeError(f"Fresh interpreter required; different module already loaded: {name}")
    if _FINDER is None:
        _FINDER = OriginalKernelFinder()
        sys.meta_path.insert(0, _FINDER)
    return _FINDER

def load_kernels():
    install()
    names = ("a9_engine", "experiment_support", "a10_common", "galerkin_rom", "operators",
             "selectors_current", "selector_coordinates", "scenes", "endpoint_batch",
             "removal_core_batch", "online_tolerance", "verified_exchange",
             "original_exchange_helpers", "original_exchange_controller")
    return {name: importlib.import_module(name) for name in names}

if __name__ == "__main__":
    import json
    modules = load_kernels()
    print(json.dumps({"status": "IMPORT_ONLY", "physical_actions": 0,
                      "old_runners_loaded": [name for name in BLOCKED if name in sys.modules],
                      "modules": {name: str(Path(module.__file__).resolve())
                                  for name, module in modules.items()}}, indent=2))
