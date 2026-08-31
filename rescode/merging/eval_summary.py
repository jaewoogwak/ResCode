"""Summarise lm-eval results across the 8 zero-shot tasks.

Computes two aggregate numbers and prints them at the end of both stdout
(captured by `tee` into the log) and the result-text file:

    avg      = unweighted mean of `acc`      across all requested tasks
    avg_norm = unweighted mean of `acc_norm` across all requested tasks

Tasks that do not report `acc_norm` (e.g. winogrande, boolq, mmlu, rte)
contribute to `avg_norm` via a fallback to `acc` so the number stays
comparable to the standard headline metric. The fallback is flagged in the
per-task table.
"""

from typing import Dict, List, Optional, Tuple


# Metrics in lm-eval are stored under f"{name},{filter}". With the default
# filter the suffix is `,none`. We try the suffixed form first and fall back
# to the bare form for older lm-eval versions.
_PRIMARY = ("acc,none", "acc")
_NORM = ("acc_norm,none", "acc_norm")


def _split_task_list(task: object) -> List[str]:
    if isinstance(task, str):
        return [t.strip() for t in task.split(",") if t.strip()]
    return [str(t) for t in task]


def _lookup_metric(metrics: Dict, candidates) -> Optional[float]:
    for k in candidates:
        if k in metrics:
            try:
                return float(metrics[k])
            except (TypeError, ValueError):
                return None
    return None


def _find_task_metrics(results_payload: Dict, task_name: str) -> Optional[Dict]:
    """Find the metrics dict for `task_name` in lm-eval's payload.

    For group tasks (e.g. mmlu) the headline numbers live under either
    `results["groups"][task_name]` or `results["results"][task_name]`,
    depending on the lm-eval version. We check both."""
    groups = results_payload.get("groups") or {}
    if task_name in groups:
        return groups[task_name]
    res = results_payload.get("results") or {}
    if task_name in res:
        return res[task_name]
    # Some older lm-eval releases prepend "lm_eval.tasks." or similar; try a
    # case-insensitive fallback.
    for k, v in res.items():
        if str(k).lower() == task_name.lower():
            return v
    for k, v in groups.items():
        if str(k).lower() == task_name.lower():
            return v
    return None


def summarize_task_metrics(
    results_payload: Dict,
    task_list: object,
) -> Dict:
    """Pull per-task acc / acc_norm and compute avg / avg_norm.

    Returns:
        {
            "tasks": [(task_name, acc, acc_norm_or_none), ...],
            "avg": float | None,
            "avg_norm_strict": float | None,    # only tasks with acc_norm
            "avg_norm": float | None,           # acc_norm if exists else acc
            "avg_norm_uses_fallback": [task_name, ...],
        }
    """
    tasks = _split_task_list(task_list)
    rows: List[Tuple[str, Optional[float], Optional[float]]] = []
    accs: List[float] = []
    norm_strict: List[float] = []
    norm_with_fb: List[float] = []
    fallback_tasks: List[str] = []

    for t in tasks:
        m = _find_task_metrics(results_payload, t)
        if m is None:
            rows.append((t, None, None))
            continue
        acc = _lookup_metric(m, _PRIMARY)
        acc_norm = _lookup_metric(m, _NORM)
        rows.append((t, acc, acc_norm))
        if acc is not None:
            accs.append(acc)
        if acc_norm is not None:
            norm_strict.append(acc_norm)
            norm_with_fb.append(acc_norm)
        elif acc is not None:
            norm_with_fb.append(acc)
            fallback_tasks.append(t)

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else None

    return {
        "tasks": rows,
        "avg": _mean(accs),
        "avg_norm_strict": _mean(norm_strict),
        "avg_norm": _mean(norm_with_fb),
        "avg_norm_uses_fallback": fallback_tasks,
    }


def format_summary(summary: Dict) -> str:
    """Render the summary as a markdown-ish text block."""
    lines = []
    lines.append("|       Task        |  acc   | acc_norm |")
    lines.append("|-------------------|-------:|---------:|")
    for name, acc, acc_norm in summary["tasks"]:
        a = f"{acc:.4f}" if acc is not None else "  -   "
        n = f"{acc_norm:.4f}" if acc_norm is not None else "    -   "
        lines.append(f"| {name:<17} | {a} | {n} |")
    lines.append("")
    avg = summary["avg"]
    avg_norm = summary["avg_norm"]
    avg_norm_strict = summary["avg_norm_strict"]
    fb = summary["avg_norm_uses_fallback"]

    def _fmt(x):
        return f"{x:.4f}" if x is not None else "n/a"

    lines.append(f"avg      = {_fmt(avg)}      (mean of acc over all tasks)")
    lines.append(
        f"avg_norm = {_fmt(avg_norm)}      "
        f"(mean of acc_norm; fallback to acc for {fb})"
        if fb else
        f"avg_norm = {_fmt(avg_norm)}      (mean of acc_norm over all tasks)"
    )
    if fb:
        lines.append(
            f"avg_norm_strict = {_fmt(avg_norm_strict)}  "
            f"(mean of acc_norm over only tasks that report acc_norm: "
            f"{[name for name, _, an in summary['tasks'] if an is not None]})"
        )
    return "\n".join(lines)


def report_eval_summary(
    results_payload: Dict,
    task_list: object,
    result_path: Optional[str] = None,
    header: str = "RESCODE evaluation summary",
) -> Dict:
    """Print summary to stdout and append to result_path. Returns the summary
    dict so callers can also log/upload it elsewhere."""
    summary = summarize_task_metrics(results_payload, task_list)
    body = format_summary(summary)
    block = (
        "\n" + ("=" * 72) + "\n"
        + header + "\n"
        + ("=" * 72) + "\n"
        + body + "\n"
        + ("=" * 72) + "\n"
    )
    print(block)
    if result_path:
        with open(result_path, "a") as f:
            f.write(block)
    return summary
