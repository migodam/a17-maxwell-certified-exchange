# Reproduce saved closure evidence

This entry uses saved arrays and audited public tables. It does not execute Maxwell physics, require the 42 external kernels, or independently inspect unpublished command/CIM/transport monitors. The ports are new public reanalysis source, not historical execution bytes. Start in the final `closure_final/` directory. Python and NumPy are required for nonlinear saved arrays; Matplotlib is optional for saved plots. Materializer, coverage and fee validation use the standard library. Dependencies and exact tool hashes must agree with the final public manifest; publication/inclusion approval is separate.

## Verify and restore explicitly selected evidence

Set `VOLUME_DIR` to downloaded evidence ZIP volumes, `ASSET_INDEX_SHA` to the final owner's published SHA of `ALIAS_BLOB_INDEX.json`, and `EVIDENCE_ROOT` to a new, absent reader directory under an existing parent. The final index is supplied at publication; no index/archive SHA is invented here.

```sh
python reanalysis/portable_v2/materialize_evidence.py --index ALIAS_BLOB_INDEX.json --index-sha256 "$ASSET_INDEX_SHA" --volumes "$VOLUME_DIR" --prefix closure_final/evidence/nonlinear
python reanalysis/portable_v2/materialize_evidence.py --index ALIAS_BLOB_INDEX.json --index-sha256 "$ASSET_INDEX_SHA" --volumes "$VOLUME_DIR" --prefix closure_final/evidence/nonlinear --materialize --out "$EVIDENCE_ROOT"
```

The first command verifies only; the second is an explicit reader-side restoration. The index schema is `a17.closure.evidence.alias.blob.index.v1`: one-based alias volume position, `public_path`, SHA, byte size, canonical `sha256/<SHA>` member path; volume filename/SHA/size/member count. The port verifies selected whole-volume SHA, exact member sets, CRC and member SHA, then restores exact aliases under `EVIDENCE_ROOT/closure_final/`. Prefixes are explicit boundary matches; no unrelated alias is selected implicitly. Existing destinations, unsafe traversal and symlinks are rejected. Do not alter immutable source/evidence bytes. A failed restoration can leave a partial new output; select another new directory rather than overwriting it.

## Recompute supported nonlinear saved arrays from two roots

The exact curated nonlinear manifest is in Git at `audits/nonlinear/PORTABLE_INPUTS.json`; arrays are in the evidence root and per-case derived curves remain in Git. Its current candidate SHA is `cb89280d9016000720f8ddf5935113a60505d03fd909535fbaeeeb9ec014b1a3`; use the final owner's manifest SHA if the actual curated manifest changes. Set `NL_MANIFEST_SHA` to that reviewed SHA.

The existing port has a single package-root interface. The following reader command joins only hash-approved manifest aliases in a new temporary layout, checks exact bytes from the two input roots and removes its temporary copies afterward. It never modifies Git files or materialized arrays; no kernels are imported.

```sh
python - "$EVIDENCE_ROOT" "$NL_MANIFEST_SHA" <<'PY'
from pathlib import Path
import hashlib,json,shutil,sys,tempfile
closure=Path.cwd().resolve()
evidence=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(closure/'reanalysis/portable_v2'))
from materialize_evidence import relative,ordinary,digest
import reanalyze_saved_nonlinear as N
manifest=ordinary(closure/'audits/nonlinear/PORTABLE_INPUTS.json')
expected=digest(sys.argv[2])
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==expected,'manifest SHA differs'
m=json.loads(manifest.read_text())
with tempfile.TemporaryDirectory(prefix='a17_saved_reader_') as temp:
    root=Path(temp).resolve()
    for name,h in m['files'].items():
        alias=relative(name);digest(h)
        choices=[evidence/alias,closure/alias.relative_to('closure_final')]
        present=[ordinary(p) for p in choices if p.exists()]
        assert present,'missing approved alias: '+name
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p in present),'input SHA differs: '+name
        target=root/alias;target.parent.mkdir(parents=True,exist_ok=True)
        with present[0].open('rb') as source,target.open('xb') as dest:
            shutil.copyfileobj(source,dest)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==h
    report,_=N.analyze(root,manifest,expected)
    assert report['counts']==dict(planned=8,completed=6,failed_prefix=1,not_run=1)
    assert sum(len(c['objective_curve']) for c in report['cases'])==125
    print(json.dumps(report,indent=2,allow_nan=False))
PY
```

This actual saved dataset supports six completed endpoints, one 17-update failed prefix and one not-run baseline. The completed pairs' 36 metrics and 18 ratios were checked against owner statistics with exact equality in `audits/nonlinear/PUBLIC_SAVED_COMPARISON.json`; six complex full-volume CSV values and 125 saved curve rows also agree. These are local saved-evidence checks, not new physics or a production certificate.

The 2016 failed prefix lacks saved measured and held fields, so measured relative error, heldout relative error and final objective remain null. `--require-full-metrics` on the unchanged full eight-case manifest intentionally fails with named missing prerequisites for that prefix. Do not drop the failure or manufacture fields to turn it into a pass. Its last-safe material diagnostics are not a completed endpoint. Saved chi slices/objective curves can be plotted from supported data using the existing port's `--write-derived --out NEW_OUTPUT --plots` in a complete single-root reader layout. No forward/held receiver operator is applied.

## Validate public coverage authority and aggregates

The current reviewed authority SHA is `e4f7c13169e570d49d6b5e0e16e8111c9c94e891dd419cded826c2652c40d978`, bound to snapshot receipt SHA `0d1463078905fcdc54dcc215d47a34e6a1629629691439805c44effeb966f7e8`.

```sh
python reanalysis/portable_v2/validate_public_coverage.py --authority audits/coverage/coverage_evidence.json --authority-sha256 e4f7c13169e570d49d6b5e0e16e8111c9c94e891dd419cded826c2652c40d978 --initial-csv audits/coverage/coverage_object_initial_opportunity.csv --path-csv audits/coverage/coverage_path_final_risk.csv --receipt-sha256 0d1463078905fcdc54dcc215d47a34e6a1629629691439805c44effeb966f7e8
```

This validates exact CSV/authority bytes, the 11-object/99-cell denominator, initial positive-full-teacher-Q weighted opportunity and separate bounded path final risk. Public cell records support recomputation of aggregates. It does not re-audit private source binding, acquisition or monitors. No global oracle or deterministic certificate is supplied. The independent saved-array review in `audits/coverage_array/REPORT.md` states which selected gains/tolerances and dense-J gaps were recomputed and which all-candidate quantities lack saved directions.

## Reconstruct public fee totals

The safe public cost authority is an audited derivative, schema `a17.public.cost.derivation.v1`, with original private source JSON SHA but no private command originals. The current candidate SHA is `fd9d842a2fbd63309480dded35223bcaaeffca3fd1cd017297f216a9a2e02955`.

```sh
python reanalysis/cost/validate_public_fee_csv.py --attempt-csv public_cost/attempt_occupation.csv --preflight-csv public_cost/rejected_preflight_fees.csv --public-authority public_cost/PUBLIC_COST_DERIVATION.json --public-authority-sha256 fd9d842a2fbd63309480dded35223bcaaeffca3fd1cd017297f216a9a2e02955
```

It reconstructs carry + whole attempt occupation + rejected-preflight totals, retaining failed charges and exact phase sums. Nested stage/counter/actual-RHS views are not added as fees. Physical acquisition/cached-label/RHS null reasons remain in the safe derivative and `nested_actual_rhs.csv`. Independent strict private monitor audit is unavailable publicly; this command checks reviewed CSV fees only.

## Source and smoke checks

```sh
python -m unittest discover -s reanalysis/portable_v2/tests -v
python -m unittest discover -s reanalysis/cost/tests -v
```

These are synthetic mechanical tests, not scientific validation. Byte SHA in the final public manifest is the public source-integrity check. Root additionally used exact original/public SHA plus AST/MatMul comparison; that check needs the private local approved plan and original owned source paths, which the public package does not supply. Runtime receipt identities and the 42 external hashes describe provenance/full-physics prerequisites; they do not redistribute vendor code or establish historical correctness by themselves.
