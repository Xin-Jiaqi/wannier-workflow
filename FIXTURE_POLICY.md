# Test-fixture policy

The repository uses three fixture tiers so a clean clone remains testable without turning an unpublished study into a public data release.

## Tier A — repository fixtures

- Synthetic, compact, repository-owned inputs only.
- Stored under `tests/fixtures/` with a machine-readable manifest.
- Included in ordinary unit tests and GitHub Actions.
- Never used as scientific evidence.

## Tier B — redistributable reference fixtures

- Small derived files or structures whose source, version, licence, transformation, and SHA256 are recorded.
- Admitted only after rights and manuscript-disclosure review.
- Prefer a versioned GitHub Release or a DOI-backed archive when files are too large for Git.
- Tests consume a downloaded, checksum-verified bundle; they do not silently fetch mutable URLs.

## Tier C — private research fixtures

- Licensed VASP inputs, POTCAR data, full database snapshots, unpublished candidate pools, and manuscript results.
- Kept outside Git and selected with `W90FLOW_REFERENCE_ROOT=/absolute/path`.
- May run local regression tests but never becomes a required dependency of public CI.

## Non-negotiable exclusions

No VASP source/binary, POTCAR, API key, private database, or unreviewed research result may be committed. `scripts/audit_repository.py` is a fast guard, not a substitute for human rights and disclosure review.
