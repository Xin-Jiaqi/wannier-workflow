# Stage-specific numerical acceptance

A completed process is not a converged model. `convergence_gate` requires normal completion and positive localization evidence; disentanglement plans additionally require positive disentanglement evidence. Explicit stage failure is rejected. Missing evidence returns `CONVERGENCE_UNVERIFIED`; exact plans do not require a disentanglement stage.

The same gate is used by numerical execution, reuse import, candidate quality evaluation, and baseline quality checks. Nonzero execution/missing-output failures retain their classification before scientific convergence is checked. Reuse refuses a source before copying its numerical outputs. Candidate quality evaluation rejects unverified old PASS records before reading DFT/plotting, and writes a rejected quality record.

`parse_wout` stage fields are evidence, not a substitute for the plan-aware gate. Historical `converged` booleans cannot be used as a global success rate. Old rankings/exports require explicit re-audit; this change does not rewrite their scientific history or imply full-pool convergence. Legacy logs without sufficient stage markers need manual/version-specific examination rather than silent acceptance.

Band-path identity, target-window comparability, model completeness and downstream response validation remain separate requirements. The WSL September development line and this software package are not otherwise fully merged by this patch.
