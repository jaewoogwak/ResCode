# Exponent-Mantissa Base + QJL, D factor 2

- `full/` contains the first compression diagnostics and `eval_all/` (ARC Easy and ARC Challenge).
- `full/rebuilt/` contains the second compression diagnostics. Its evaluation files are in `full/eval_8tasks/`.
- `full/eval_8tasks/` contains these completed tasks: arc_easy, arc_challenge, mmlu, openbookqa, rte. The directory name does not indicate eight completed tasks.
- `full/compact_qjl_state.pt` is the rebuilt checkpoint on this server. It exceeds GitHub ordinary Git limits and is not committed.
- The two compression runs have identical aggregate Frobenius statistics and Base/QJL tensor byte counts. Timing records are kept separately.
