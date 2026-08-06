# Preprocess prototype handoff

Source snapshot copied from:

`/private/tmp/w90flow-stage1-generalization-20260722`

Isolated working tree:

`/private/tmp/w90flow-preprocess-agent-20260722`

## Preprocess-owned files

- `src/w90flow/preprocess.py`
- `tests/test_preprocess.py`
- `tests/fixtures/preprocess/README.md`
- `tests/fixtures/preprocess/2d-relax-pass/{POSCAR,CONTCAR,OUTCAR,OSZICAR}`
- `PREPROCESS_CONTRACT.md`
- `PREPROCESS_HANDOFF.md`
- `examples/preprocess.example.yaml`

The executable finite-state extension remains confined to these preprocess-owned
files.  It does not edit candidate generation, PyW90, attempt planning, ranking,
or the existing VASP/Wannier executor.

## Verification

Run from this isolated working tree:

```text
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
```

The preprocess suite includes a tiny synthetic Python child to exercise the
explicit execute path.  No VASP/Wannier calculation is launched.

## Recommended integration order

1. Copy the new module, tests, fixture, documentation and example only.
2. Re-run the current Stage 1 suite because the source snapshot is concurrently evolving.
3. The prepare/run/inspect/recover/reconcile CLI wrappers now consume the checked-in
   `preprocess_inputs` schema and still default execution to dry-run.
4. Keep actual execution behind an explicit `--execute` permission gate.
5. Validate parsers against sanitized real outputs from at least one 2D ions-only,
   one 2D constrained-cell, and one 3D full-cell case before enabling production.
6. Connect a PASS Band handoff to the existing `fit-from-bands` entrypoint by exact hashes.
