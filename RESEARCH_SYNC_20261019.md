# Private research sync — 2026-10-19

Stage 1 includes the previously local autonomous runner, platform preflight, validated reuse, k-point/neighbour checks, accepted checkpoints and bounded recovery proposals. The WSL September branch and October strict runtime are separate implementations retained in the private research archive; this sync does not claim they have been integrated or that the full material pool passed strict convergence.

This branch combines pre-existing local commits with the current remote security/CI fixes, preserving their history. It is an archival/review checkpoint, not a new production release. No success-rate claim is introduced. See the private Xin-Jiaqi/wannauto-research-archive repository for dated project documents, cohort records and the separate runtime sources.

## Strict acceptance follow-up

The stage-aware gate has now been ported into this branch (10d60e4, with persisted rejection state in43bbc8c), covering execution, reuse and both quality entry points. 270 tests run with4 private-fixture skips on Mac and WSL; Python3.10/3.12 CI passes. Three historical candidates have been replayed with native-path verification; the full48-case retrospective log audit is documented in the private archive. This partial integration does not imply all September functionality, old rankings/checkpoints, or scientific results have been revalidated. See STRICT_ACCEPTANCE.md.
