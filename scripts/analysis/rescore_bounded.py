"""Rescore every power-check cell on both WER definitions, from dev_rows.

`atcgen/rl/reward.py` changed at 02:43 to score the reward on a *bounded* WER
(each row's errors capped at its reference word count).  Cells run before that
recorded the unbounded number, cells run after recorded the bounded one, and
`results.jsonl` therefore mixes two metrics.  The per-row files carry raw
uncapped counts on purpose, so both definitions can be recomputed for every
cell and compared on equal terms without rerunning anything.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DEFAULT_RUN = Path("runs/power_check_kixd")


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def score(rs: list[dict]) -> tuple[float, float, int]:
    """(unbounded WER, bounded WER, rows whose errors exceeded their length)."""
    e = sum(r["errors"] for r in rs)
    w = sum(r["ref_words"] for r in rs)
    cap = sum(min(r["errors"], r["ref_words"]) for r in rs)
    n = sum(1 for r in rs if r["errors"] > r["ref_words"])
    return e / w, cap / w, n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="?", default=str(DEFAULT_RUN),
                    help="power-check run dir with harness/ and trials/ (default %(default)s)")
    run = Path(ap.parse_args(argv).run)
    base_rows = rows(next((run / "harness" / "baseline").glob("*_rows.jsonl")))
    b_un, b_bd, b_n = score(base_rows)
    print(f"zero-shot baseline: unbounded {b_un:.4f}  bounded {b_bd:.4f}  "
          f"({b_n} rows capped)\n")

    cells = {}
    for f in sorted(run.glob("trials/*/dev_rows.jsonl")):
        cells[f.parent.name] = rows(f)

    print(f"{'cell':<14}{'WER_unb':>9}{'WER_bnd':>9}{'capped':>8}"
          f"{'R_unb':>9}{'R_bnd':>9}")
    per_arm = {}
    for name, rs in sorted(cells.items()):
        un, bd, n = score(rs)
        r_un, r_bd = b_un - un, b_bd - bd
        arm = name.rsplit("_s", 1)[0]
        per_arm.setdefault(arm, []).append((r_un, r_bd))
        print(f"{name:<14}{un:>9.4f}{bd:>9.4f}{n:>8}{r_un:>+9.4f}{r_bd:>+9.4f}")

    print(f"\n{'arm':<12}{'n':>3}{'mean R_unb':>12}{'sd':>9}"
          f"{'mean R_bnd':>12}{'sd':>9}")
    summ = {}
    for arm, vals in sorted(per_arm.items()):
        un = [v[0] for v in vals]
        bd = [v[1] for v in vals]
        s_un = statistics.stdev(un) if len(un) > 1 else float("nan")
        s_bd = statistics.stdev(bd) if len(bd) > 1 else float("nan")
        summ[arm] = (statistics.mean(un), s_un, statistics.mean(bd), s_bd, len(un))
        print(f"{arm:<12}{len(un):>3}{statistics.mean(un):>+12.4f}{s_un:>9.4f}"
              f"{statistics.mean(bd):>+12.4f}{s_bd:>9.4f}")

    if "base" in summ:
        print("\ngap vs base, in units of that arm-pair's pooled sd:")
        bm_un, bs_un, bm_bd, bs_bd, bn = summ["base"]
        for arm, (m_un, s_un, m_bd, s_bd, n) in sorted(summ.items()):
            if arm == "base":
                continue
            p_un = (((bn - 1) * bs_un ** 2 + (n - 1) * s_un ** 2)
                    / max(bn + n - 2, 1)) ** 0.5
            p_bd = (((bn - 1) * bs_bd ** 2 + (n - 1) * s_bd ** 2)
                    / max(bn + n - 2, 1)) ** 0.5
            print(f"  {arm:<12} unbounded {m_un - bm_un:+.4f} "
                  f"({abs(m_un - bm_un) / p_un:.1f}x)   "
                  f"bounded {m_bd - bm_bd:+.4f} ({abs(m_bd - bm_bd) / p_bd:.1f}x)")


if __name__ == "__main__":
    main()
