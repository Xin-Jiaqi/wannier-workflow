# Stage-specific numerical acceptance

A completed process is not a converged model. `convergence_gate` requires normal completion and positive localization evidence; disentanglement plans additionally require positive disentanglement evidence. Explicit stage failure is rejected. Missing evidence returns `CONVERGENCE_UNVERIFIED`; exact plans do not require a disentanglement stage.

The same gate is used by numerical execution, reuse import, candidate quality evaluation, and baseline quality checks. Nonzero execution/missing-output failures retain their classification before scientific convergence is checked. Reuse refuses a source before copying its numerical outputs. Candidate quality evaluation rejects unverified old PASS records before reading DFT/plotting, and writes a rejected quality record.

`parse_wout` stage fields are evidence, not a substitute for the plan-aware gate. Historical `converged` booleans cannot be used as a global success rate. Old rankings/exports require explicit re-audit; this change does not rewrite their scientific history or imply full-pool convergence. Legacy logs without sufficient stage markers need manual/version-specific examination rather than silent acceptance.

Band-path identity, target-window comparability, model completeness and downstream response validation remain separate requirements. The WSL September development line and this software package are not otherwise fully merged by this patch.

## Opt-in single-factor disentanglement control (2026-10-19)

`iteration.disentanglement_budget_iterations: 2000` adds **one** finite attempt
per disentanglement candidate, selected only after an explicit
`DISENTANGLEMENT_NOT_CONVERGED` result. The default `0` adds no attempt.
Accepted values are integer 1000–10000, excluding booleans. This is an additional
budget beyond the configured window/localization retries; account for its runtime.
It copies the seed plan and changes only `dis_num_iter`, retaining windows,
projection, mixing, localization settings and all quality thresholds. Exact
candidates receive no extra trial. Missing convergence evidence and fit-only
failures do not select this branch. Once consumed, the existing finite policy
may choose a different available retry; it cannot regenerate this trial.

Evidence motivating the option: Li2C2's fixed-input 500→2000 control converged
with max band error 3.639 meV in its declared evaluation window; this is one
material, not a universal recovery guarantee. The new scheduler integration has
unit/regression validation only until an isolated real run is completed.
