"""Paired-by-seed comparison of every power-check arm against base.

Every arm is evaluated at the same seeds as `base`, and a seed fixes both the
generator draw and the fine-tune batch order -- so seed is a shared nuisance
factor and pairing removes it.  The unpaired "separation" figure the runner
prints does not, which is why it reads weaker than the evidence warrants.

All rewards are recomputed from dev_rows on the bounded WER (each row's errors
capped at its reference length), so cells run before and after the 02:43 reward
change are on one metric.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

RUN = Path("runs/power_check_kixd")


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def bounded(rs: list[dict]) -> float:
    w = sum(r["ref_words"] for r in rs)
    return sum(min(r["errors"], r["ref_words"]) for r in rs) / w


def main() -> None:
    base_wer = bounded(rows(next((RUN / "harness" / "baseline").glob("*_rows.jsonl"))))
    cell = {f.parent.name: base_wer - bounded(rows(f))
            for f in RUN.glob("trials/*/dev_rows.jsonl")}

    arms, seeds = {}, set()
    for name in cell:
        arm, _, s = name.rpartition("_s")
        arms.setdefault(arm, {})[int(s)] = cell[name]
        seeds.add(int(s))

    order = ["base", "aug_off", "speed_fixed", "voiceaug_off", "pitch_off", "degraded"]
    order = [a for a in order if a in arms] + [a for a in arms if a not in order]
    seeds = sorted(seeds)

    print(f"bounded zero-shot baseline WER {base_wer:.4f}\n")
    print("reward by arm and seed (bounded)")
    print(f"{'arm':<14}" + "".join(f"{'s'+str(s):>10}" for s in seeds)
          + f"{'mean':>10}{'sd':>9}")
    for a in order:
        v = arms[a]
        vals = [v[s] for s in seeds if s in v]
        print(f"{a:<14}"
              + "".join(f"{v[s]:>+10.4f}" if s in v else f"{'-':>10}" for s in seeds)
              + f"{st.mean(vals):>+10.4f}"
              + (f"{st.stdev(vals):>9.4f}" if len(vals) > 1 else f"{'-':>9}"))

    print("\npaired against base at shared seeds "
          "(positive = base better, i.e. the arm HURTS)")
    print(f"{'arm':<14}{'n':>3}{'mean':>9}{'sd':>9}{'se':>9}{'t':>7}{'df':>4}"
          f"{'dir':>7}   per-seed diffs")
    for a in order:
        if a == "base":
            continue
        sh = sorted(set(arms[a]) & set(arms["base"]))
        d = [arms["base"][s] - arms[a][s] for s in sh]
        n = len(d)
        m = st.mean(d)
        sd = st.stdev(d) if n > 1 else float("nan")
        se = sd / n ** 0.5 if n > 1 else float("nan")
        print(f"{a:<14}{n:>3}{m:>+9.4f}{sd:>9.4f}{se:>9.4f}{m / se:>7.2f}{n - 1:>4}"
              f"{sum(1 for x in d if x > 0):>4}/{n}   "
              + " ".join(f"{x:+.4f}" for x in d))

    if {"speed_fixed", "voiceaug_off", "aug_off"} <= set(arms):
        print("\nadditivity check -- speed_fixed and voiceaug_off partition aug_off")
        parts = {}
        for a in ("speed_fixed", "voiceaug_off", "aug_off"):
            sh = sorted(set(arms[a]) & set(arms["base"]))
            parts[a] = st.mean([arms["base"][s] - arms[a][s] for s in sh])
        tot = parts["speed_fixed"] + parts["voiceaug_off"]
        print(f"  speed_fixed  {parts['speed_fixed']:+.4f}"
              f"   ({100 * parts['speed_fixed'] / tot:.0f}% of the two halves)")
        print(f"  voiceaug_off {parts['voiceaug_off']:+.4f}"
              f"   ({100 * parts['voiceaug_off'] / tot:.0f}%)")
        print(f"  sum          {tot:+.4f}")
        print(f"  aug_off      {parts['aug_off']:+.4f}   "
              f"(residual {tot - parts['aug_off']:+.4f})")


if __name__ == "__main__":
    main()
