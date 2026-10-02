# Public audit presentation

Source/runtime and CSV lineage come from the original provenance audit. JSON pointers are represented as `pointer_tokens`, the RFC 6901 decoded token array, with an `original_pointer_sha256`. Re-escape each token (`~` to `~0`, `/` to `~1`) and join with slashes to recover the exact original pointer. Full reversal fingerprints and original audit hashes are in PUBLIC_AUDIT_PROJECTION_RECEIPT.json. No observations, rows, identities, numeric values, or references were dropped. This is a metadata presentation change; primary Maxwell JSON and source matrices follow their own immutable manifests.

The independently authored CPU algebra test is a separate nonhistorical version without an interpreter header. Its AST matches the draft test. All eligible executed A17 Python files are copied as original bytes; they are not routed through this projection. Unobserved analysis/test files remain labeled as unobserved rather than being assigned an execution receipt.
