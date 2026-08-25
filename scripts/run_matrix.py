"""Run the research-integration validation matrix (docs/plans/research-integration.md).

Arms (whisper-tiny.en student, budget-matched SFT steps, frozen normalization):
  A0   zero-shot
  A1   SFT on real_train (jacktol train[0:8000])
  A2   SFT on gate-selected synthetic (gold+silver+adversarial<=5%)
  A2u  SFT on the same synthetic pool, ungated (gate-value ablation)
  A3   SFT on mix (default 75% real / 25% gated synthetic)
  A4   A3 recipe + GRPO stage

Stages are resumable: each is skipped when its output artifact already exists,
so rerunning the same command continues after a crash. Model selection uses
model_select (train[9000:10000]); locked_test (test[500:2500]) is read once per
arm at the very end (D11) and paired bootstrap compares the headline arms.

  uv run python scripts/run_matrix.py --out runs/matrix_v1
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SFT_ARMS = {
    # name -> extra recipe args
    "a1_real": ["--arm", "real_only"],
    "a2_synth_gated": ["--arm", "synth_only"],
    "a2u_synth_ungated": ["--arm", "synth_only"],
    "a3_mix": ["--arm", "mix"],
    "a4_mix_grpo": ["--arm", "mix_grpo"],
}


def sh(args: list[str], log: Path) -> None:
    print(f"+ {' '.join(str(a) for a in args)}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n+ {' '.join(str(a) for a in args)}\n")
        f.flush()
        proc = subprocess.run([str(a) for a in args], stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = "".join(open(log).readlines()[-30:])
        raise RuntimeError(f"command failed ({proc.returncode}): {args}\n--- log tail ---\n{tail}")


def py(module_or_script: str, *args: str) -> list[str]:
    return [sys.executable, str(REPO / module_or_script), *args]


def stage_generate(out: Path, n_synth: int, seed: int, config: str, text: str, log: Path) -> Path:
    pool = out / "synth_pool"
    manifest = pool / "manifest.jsonl"
    if manifest.exists() and sum(1 for _ in open(manifest)) >= n_synth:
        print(f"[gen] exists: {manifest}")
        return pool
    sh(py("scripts/generate_dataset.py", "--config", config, "--n-samples", str(n_synth),
          "--out", str(pool), "--seed", str(seed), "--text", text), log)
    return pool


def stage_gate(pool: Path, log: Path) -> Path:
    gated = pool / "manifest_gated.jsonl"
    if gated.exists():
        print(f"[gate] exists: {gated}")
        return gated
    sh(py("scripts/gate_dataset.py", "--dataset", str(pool)), log)
    return gated


def stage_select(pool: Path, gated: Path) -> Path:
    selected = pool / "manifest_selected.jsonl"
    if selected.exists():
        print(f"[select] exists: {selected}")
        return selected
    from atcgen.gate.gate import select_tiers
    rows = select_tiers(gated, tiers=("gold", "silver", "adversarial"), adversarial_cap=0.05)
    with open(selected, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[select] kept {len(rows)} rows -> {selected}")
    return selected


def stage_arm(name: str, extra: list[str], out: Path, args, pool: Path,
              selected: Path, log_dir: Path) -> dict:
    arm_dir = out / "arms" / name
    run_json = arm_dir / "run.json"
    if run_json.exists():
        print(f"[arm {name}] exists")
        return json.loads(run_json.read_text())
    synth = selected if name != "a2u_synth_ungated" else pool / "manifest.jsonl"
    cmd = py("training/recipe.py",
             "--out", str(arm_dir), "--model", args.model,
             "--real-split", "train", "--real-indices", "0:8000",
             "--dev-split", "train", "--dev-indices", "9000:9400",
             "--sft-steps", str(args.sft_steps), "--sft-batch", str(args.sft_batch),
             "--sft-lr", str(args.sft_lr), "--mix-ratio", str(args.mix_ratio),
             "--grpo-steps", str(args.grpo_steps), "--grpo-lr", str(args.grpo_lr),
             "--seed", str(args.seed), *extra)
    if name != "a1_real":
        cmd += ["--synth-manifest", str(synth)]
    sh(cmd, log_dir / f"{name}.log")
    return json.loads(run_json.read_text())


def stage_eval(tag: str, model: str, split: str, out: Path, log_dir: Path) -> dict:
    report_path = out / "eval" / f"{tag}_{split}.json"
    if report_path.exists():
        print(f"[eval {tag}/{split}] exists")
        return json.loads(report_path.read_text())
    sh(py("training/evaluate.py", "--model", model, "--split-name", split,
          "--report-out", str(report_path),
          "--hyps-out", str(out / "eval" / f"{tag}_{split}_hyps.jsonl"),
          "--batch-size", "8"), log_dir / f"eval_{tag}_{split}.log")
    return json.loads(report_path.read_text())


def paired(out: Path, split: str, tag_a: str, tag_b: str) -> dict | None:
    """Paired bootstrap; delta = WER(tag_a) - WER(tag_b), so delta > 0 means B wins."""
    from atcgen.rl.stats import paired_bootstrap
    path_a = out / "eval" / f"{tag_a}_{split}_hyps.jsonl"
    path_b = out / "eval" / f"{tag_b}_{split}_hyps.jsonl"
    if not (path_a.exists() and path_b.exists()):
        return None
    rows_a = [json.loads(line) for line in open(path_a)]
    rows_b = [json.loads(line) for line in open(path_b)]
    refs = [r["reference"] for r in rows_a]
    return paired_bootstrap(refs, [r["hypothesis"] for r in rows_a],
                            [r["hypothesis"] for r in rows_b])


def summarize(out: Path, reports: dict[str, dict], split: str) -> dict:
    def pick(tag):
        r = reports.get(tag)
        if not r:
            return None
        e = r.get("entities") or {}
        return {
            "wer": r["wer"]["atc_normalized"], "wer_raw": r["wer"]["raw"],
            "substitutions": r["wer"].get("substitutions"),
            "deletions": r["wer"].get("deletions"),
            "insertions": r["wer"].get("insertions"),
            "callsign_accuracy": (e.get("callsign") or {}).get("accuracy"),
            "entity_f1": (e.get("overall") or {}).get("f1"),
            "entity_recall": (e.get("overall") or {}).get("recall"),
            "critical_substitution_rate": (e.get("critical") or {}).get("substitution_rate"),
        }
    rows = {tag: pick(tag) for tag in reports}
    a1 = rows.get("a1_real") or {}
    a2 = rows.get("a2_synth_gated") or {}
    verdicts = {}
    if a1 and a2 and a1.get("wer") is not None:
        gap = a2["wer"] - a1["wer"]
        verdicts["p2_synth_vs_real_gap_abs_wer"] = round(gap, 4)
        verdicts["p2_parity_bar_1p5_met"] = bool(gap <= 0.015)
    for name, (x, y) in {"p3_mix_beats_real": ("a3_mix", "a1_real"),
                         "p4a_grpo_beats_sft": ("a4_mix_grpo", "a3_mix"),
                         "gate_earns_cost": ("a2_synth_gated", "a2u_synth_ungated")}.items():
        rx, ry = rows.get(x) or {}, rows.get(y) or {}
        if rx.get("wer") is not None and ry.get("wer") is not None:
            verdicts[name] = bool(rx["wer"] < ry["wer"])
    stats = {}
    for label, (x, y) in {"a3_vs_a1": ("a1_real", "a3_mix"),
                          "a4_vs_a3": ("a3_mix", "a4_mix_grpo"),
                          "a2_vs_a2u": ("a2u_synth_ungated", "a2_synth_gated")}.items():
        result = paired(out, split, x, y)
        if result:
            stats[label] = result
    return {"split": split, "arms": rows, "verdicts": verdicts, "paired_bootstrap": stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/matrix_v1")
    ap.add_argument("--model", default="openai/whisper-tiny.en")
    ap.add_argument("--config", default="configs/mode1_matched.yaml")
    ap.add_argument("--text", default="grammar:region=eu")
    ap.add_argument("--n-synth", type=int, default=8000)
    ap.add_argument("--gen-seed", type=int, default=101)
    ap.add_argument("--sft-steps", type=int, default=2000)
    ap.add_argument("--sft-batch", type=int, default=8)
    ap.add_argument("--sft-lr", type=float, default=1e-5)
    ap.add_argument("--mix-ratio", type=float, default=0.75)
    ap.add_argument("--grpo-steps", type=int, default=600)
    ap.add_argument("--grpo-lr", type=float, default=2e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default=",".join(SFT_ARMS),
                    help="comma-separated subset of arms to train")
    ap.add_argument("--arm-workers", type=int, default=5,
                    help="concurrent training arms (independent; share MPS)")
    ap.add_argument("--eval-workers", type=int, default=3,
                    help="concurrent evaluation subprocesses")
    ap.add_argument("--skip-final", action="store_true",
                    help="stop before the locked_test reads")
    args = ap.parse_args()

    out = Path(args.out)
    log_dir = out / "logs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    t0 = time.monotonic()

    pool = stage_generate(out, args.n_synth, args.gen_seed, args.config, args.text,
                          log_dir / "generate.log")
    gated = stage_gate(pool, log_dir / "gate.log")
    selected = stage_select(pool, gated)

    # arms are independent -> run them concurrently; MPS serializes kernels but
    # aggregate throughput beats sequential on Apple Silicon (unified memory).
    arm_names = args.arms.split(",")
    with ThreadPoolExecutor(max_workers=args.arm_workers) as pool_exec:
        futures = {name: pool_exec.submit(stage_arm, name, SFT_ARMS[name], out,
                                          args, pool, selected, log_dir)
                   for name in arm_names}
        arm_runs = {name: future.result() for name, future in futures.items()}

    # model-selection reads (tuning-legal split)
    with ThreadPoolExecutor(max_workers=args.eval_workers) as pool_exec:
        futures = {"a0_zero_shot": pool_exec.submit(
            stage_eval, "a0_zero_shot", args.model, "model_select", out, log_dir)}
        for name, run in arm_runs.items():
            futures[name] = pool_exec.submit(stage_eval, name, run["final_checkpoint"],
                                             "model_select", out, log_dir)
        dev_reports = {tag: future.result() for tag, future in futures.items()}
    dev_summary = summarize(out, dev_reports, "model_select")
    (out / "summary_model_select.json").write_text(json.dumps(dev_summary, indent=2) + "\n")
    print(json.dumps(dev_summary["arms"], indent=2))
    print(json.dumps(dev_summary["verdicts"], indent=2))

    if args.skip_final:
        print(f"done (skip-final) in {time.monotonic() - t0:.0f}s")
        return

    # the one locked_test read per arm (D11)
    with ThreadPoolExecutor(max_workers=args.eval_workers) as pool_exec:
        futures = {"a0_zero_shot": pool_exec.submit(
            stage_eval, "a0_zero_shot", args.model, "locked_test", out, log_dir)}
        for name, run in arm_runs.items():
            futures[name] = pool_exec.submit(stage_eval, name, run["final_checkpoint"],
                                             "locked_test", out, log_dir)
        final_reports = {tag: future.result() for tag, future in futures.items()}
    final_summary = summarize(out, final_reports, "locked_test")
    (out / "summary_locked_test.json").write_text(json.dumps(final_summary, indent=2) + "\n")
    print(json.dumps(final_summary, indent=2))
    print(f"matrix complete in {(time.monotonic() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
