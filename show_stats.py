#!/usr/bin/env python3
"""
show_stats.py — AI Squat Coach session history viewer.

Reads session_log.json (written by session_logger.py on every quit) and
prints a formatted progress report in the terminal:

  • Per-session summary table  (date, reps, avg score, duration, top issue)
  • Lifetime totals             (total reps, total time, all-time avg score)
  • Score trend sparkline       (ASCII bar chart across all sessions)
  • Trend direction             (improving / steady / declining)
  • Most common form issues     (horizontal bar chart with percentages)
  • Personal records            (best score session, most-reps session)

Usage
-----
    python3 show_stats.py                       # reads ./session_log.json
    python3 show_stats.py --file path/to/log    # custom log path
    python3 show_stats.py --last 5              # show only last 5 sessions
    python3 show_stats.py --last 5 --no-colour  # plain text (for piping)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from typing import Any


# ANSI colour helpers — automatically disabled when stdout is not a terminal


_USE_COLOUR = sys.stdout.isatty()


def _ansi(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text


def green(t: str) -> str:  return _ansi("32", t)
def yellow(t: str) -> str: return _ansi("33", t)
def red(t: str) -> str:    return _ansi("31", t)
def bold(t: str) -> str:   return _ansi("1",  t)
def dim(t: str) -> str:    return _ansi("2",  t)
def cyan(t: str) -> str:   return _ansi("36", t)


def score_colour(score: float) -> str:
    """Return score string coloured green / yellow / red based on quality."""
    s = f"{score:.1f}"
    if score >= 85:
        return green(s)
    if score >= 65:
        return yellow(s)
    return red(s)



# ASCII sparkline  (score values 0–100 → Unicode block characters)


_BLOCKS = " ▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 42) -> str:
    """
    Build a single-line sparkline for *values* scaled to 0–100.
    Resamples to exactly *width* columns so it always fits the terminal.
    """
    if not values:
        return dim("no data yet")

    step = max(len(values) / width, 1.0)
    chars: list[str] = []
    for i in range(min(width, len(values))):
        idx = min(int(i * step), len(values) - 1)
        v = max(0.0, min(100.0, values[idx]))
        char_idx = int(v / 100.0 * (len(_BLOCKS) - 1))
        chars.append(_BLOCKS[char_idx])
    return "".join(chars)



# Trend helper


def trend_label(values: list[float]) -> str:
    """Compare first half vs second half of score history and return a label."""
    if len(values) < 2:
        return dim("not enough data")
    mid = len(values) // 2
    first = statistics.mean(values[:mid])
    second = statistics.mean(values[mid:])
    delta = second - first
    if delta > 2.5:
        return green("▲  improving  (+{:.1f} pts avg)".format(delta))
    if delta < -2.5:
        return red("▼  declining  ({:.1f} pts avg)".format(delta))
    return yellow("→  steady")



# Duration formatter


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def fmt_date(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%d %b %Y  %H:%M")
    except ValueError:
        return iso[:19]   # fallback: just slice the ISO string



# Main report


def print_report(sessions: list[dict[str, Any]], label: str) -> None:
    if not sessions:
        print(red("  No sessions found in the log."))
        print(dim("  Complete a workout and quit (q / ESC) to generate one."))
        return

    # Header 
    rule = "═" * 70
    print()
    print(bold(cyan(rule)))
    print(bold(cyan(f"  AI Squat Coach — Session History  ({label})")))
    print(bold(cyan(rule)))

    # Per-session table 
    hdr = (
        f"  {'#':>3}  {'Date & Time':<21} {'Reps':>5}  "
        f"{'Avg score':>9}  {'Duration':>9}  {'Top issue'}"
    )
    print()
    print(bold(hdr))
    print(dim("  " + "─" * 68))

    all_scores: list[float] = []
    total_reps = 0
    total_time = 0.0
    correction_totals: dict[str, int] = {}

    # Pre-compute the overall best score for the ★ marker
    best_score_value = max((s.get("avg_score", 0.0) for s in sessions), default=0.0)

    for i, s in enumerate(sessions, 1):
        avg   = float(s.get("avg_score", 0.0))
        reps  = int(s.get("total_reps", 0))
        dur   = float(s.get("duration_seconds", 0.0))
        date  = fmt_date(s.get("date", ""))
        issue = s.get("most_frequent_issue") or "—"
        corrs = s.get("corrections_by_type", {}) or {}

        all_scores.append(avg)
        total_reps += reps
        total_time += dur
        for k, v in corrs.items():
            correction_totals[k] = correction_totals.get(k, 0) + int(v)

        # Mark the personal-best session with a star
        marker = bold(yellow(" ★")) if avg == best_score_value and reps > 0 else "  "

        # Truncate long issue names so the table stays on one line
        issue_display = issue[:24] if len(issue) > 24 else issue

        print(
            f"  {i:>3}{marker} {date:<21} {reps:>5}  "
            f"{score_colour(avg):>9}  {fmt_duration(dur):>9}  "
            f"{dim(issue_display)}"
        )

    print(dim("  " + "─" * 68))

    # Lifetime totals 
    print()
    print(bold("  LIFETIME TOTALS"))
    print(f"    Sessions   :  {bold(str(len(sessions)))}")
    print(f"    Total reps :  {bold(str(total_reps))}")
    print(f"    Total time :  {bold(fmt_duration(total_time))}")
    if all_scores:
        mean_score = statistics.mean(all_scores)
        print(f"    Avg score  :  {score_colour(mean_score)}  (all sessions)")
        print(f"    Best score :  {green(f'{max(all_scores):.1f}')}")
        print(f"    Worst score:  {red(f'{min(all_scores):.1f}')}")

    # Score sparkline 
    print()
    print(bold("  SCORE TREND  (oldest → latest)"))
    print(f"    {sparkline(all_scores)}")
    print(f"    {trend_label(all_scores)}")
    print(
        f"    {dim('0')}{'':>40}{dim('100')}"
    )

    # Form issue bar chart 
    if correction_totals:
        print()
        print(bold("  MOST COMMON FORM ISSUES  (all sessions combined)"))
        sorted_issues = sorted(correction_totals.items(), key=lambda x: x[1], reverse=True)
        grand_total   = sum(correction_totals.values()) or 1
        max_count     = sorted_issues[0][1] or 1
        bar_width     = 32

        for issue_name, count in sorted_issues:
            filled  = int(count / max_count * bar_width)
            pct     = count / grand_total * 100
            bar     = "█" * filled + "░" * (bar_width - filled)
            # Colour the bar: red if > 40 % of all corrections, yellow if > 20 %
            if pct > 40:
                bar_str = red(bar)
            elif pct > 20:
                bar_str = yellow(bar)
            else:
                bar_str = green(bar)
            print(f"    {issue_name:<26}  {bar_str}  {count:>4}x  ({pct:.0f}%)")

    # Personal records 
    print()
    print(bold("  PERSONAL RECORDS"))
    if sessions:
        best_session = max(sessions, key=lambda s: s.get("avg_score", 0.0))
        most_reps_session = max(sessions, key=lambda s: s.get("total_reps", 0))

        best_date  = fmt_date(best_session.get("date", ""))
        reps_date  = fmt_date(most_reps_session.get("date", ""))

        score = best_session.get('avg_score', 0)

        print(
            f"    Best avg score :  "
            f"{green(f'{score:.1f}')}"
            f"  on {best_date}"
    )
        print(
            f"    Most reps      :  "
            f"{bold(str(most_reps_session.get('total_reps', 0)))} reps"
            f"  on {reps_date}"
        )

    print()
    print(bold(cyan(rule)))
    print()



# CLI entry point


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="show_stats.py",
        description="View your AI Squat Coach session history.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 show_stats.py\n"
            "  python3 show_stats.py --last 10\n"
            "  python3 show_stats.py --file ~/workouts/session_log.json\n"
        ),
    )
    parser.add_argument(
        "--file", "-f",
        default="session_log.json",
        metavar="PATH",
        help="Path to session_log.json  (default: ./session_log.json)",
    )
    parser.add_argument(
        "--last", "-n",
        type=int,
        default=None,
        metavar="N",
        help="Show only the last N sessions",
    )
    parser.add_argument(
        "--no-colour",
        action="store_true",
        help="Disable ANSI colours (useful when piping output)",
    )
    args = parser.parse_args()

    # Honour --no-colour by monkeypatching the module-level flag
    if args.no_colour:
        global _USE_COLOUR          # noqa: PLW0603
        _USE_COLOUR = False

    # Validate file ─
    if not os.path.exists(args.file):
        print(red(f"\n  File not found: {args.file}"))
        print(
            dim(
                "  The log is created when you quit the coach (q / ESC).\n"
                "  Run a session first, then re-run this script.\n"
            )
        )
        sys.exit(1)

    # Parse JSON 
    try:
        with open(args.file, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, ValueError) as exc:
        print(red(f"\n  Could not parse {args.file}: {exc}\n"))
        sys.exit(1)

    if not isinstance(data, list):
        print(red("\n  Unexpected format — expected a JSON array of session objects.\n"))
        sys.exit(1)

    # Optionally slice to last N 
    sessions = data
    label    = args.file
    if args.last is not None:
        sessions = data[-args.last:]
        label = f"last {args.last} of {len(data)} sessions"

    print_report(sessions, label)


if __name__ == "__main__":
    main()
