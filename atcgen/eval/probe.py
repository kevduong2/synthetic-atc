"""Tier 2 real-vs-synthetic channel probe (docs/plans/05-evaluation-plan.md §2).

A small classifier on *frozen* WavLM embeddings tries to tell the synthetic set
from the real calibration set under k-fold cross-validation.  Near-chance
accuracy means the domains match; the acceptance gate is <=0.65, good is
<=0.55.  A high-accuracy probe also localizes the gap: `layer_sweep` reports
accuracy per WavLM layer from a single pass of forward passes, and WavLM layers
are known to encode channel information (2501.05310), so the layer profile says
*where* the mismatch lives.

The classifier is a plain logistic regression (optionally one hidden layer)
trained with torch -- L2 via the optimizer's weight decay, features standardized
on each training fold.  No sklearn: the model is four lines and the fold
plumbing needs to be deterministic and inspectable anyway.

768 features against ~100 clips per class is a regime where a classifier can
separate almost anything, so an accuracy is only readable next to its empirical
chance floor: `null_control` probes one set against a random half-split of
itself, and `probe_dirs`/`layer_sweep` report it by default.  The L2 default is
set where that floor sits at ~0.52 (worst split 0.59) on the real calibration
set while true real-vs-synthetic separation is untouched.

CLI: python -m atcgen.eval.probe <syn_dir> <real_dir> [--sweep] [--out p.json]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .embed_dist import WAVLM_LAYER, embed_clips, wavlm_embedder

FOLDS = 5
L2 = 1.0            # optimizer weight decay; see the null-floor note above
L2_MLP = 0.05       # the same decay collapses a two-layer net to a constant
EPOCHS = 300
LR = 0.05
SWEEP_LAYERS = tuple(range(13))     # 0 = CNN output, 1-12 = transformer layers
GATE_ACCEPT = 0.65                  # 05 §3 acceptance criteria
GATE_GOOD = 0.55


def _stratified_folds(y: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    """Test-index arrays for `folds` class-balanced splits."""
    rng = np.random.default_rng(seed)
    chunks: list[list[np.ndarray]] = [[] for _ in range(folds)]
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        for k, part in enumerate(np.array_split(idx, folds)):
            chunks[k].append(part)
    return [np.concatenate(c) for c in chunks]


def _fit_predict(x_tr, y_tr, x_te, l2: float, epochs: int, lr: float,
                 hidden: int | None, seed: int) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
    x_tr_t = torch.tensor((x_tr - mu) / sd, dtype=torch.float32)
    x_te_t = torch.tensor((x_te - mu) / sd, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

    d = x_tr.shape[1]
    model = (nn.Linear(d, 1) if hidden is None else
             nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1)))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(x_tr_t).squeeze(-1), y_tr_t).backward()
        opt.step()
    with torch.no_grad():
        return (model(x_te_t).squeeze(-1).numpy() > 0.0).astype(np.float64)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class recall: chance is 0.5 whatever the class ratio is."""
    recalls = [float((y_pred[y_true == c] == c).mean())
               for c in np.unique(y_true) if (y_true == c).any()]
    return float(np.mean(recalls))


def probe(x_syn: np.ndarray, x_real: np.ndarray, folds: int = FOLDS,
          l2: float | None = None, epochs: int = EPOCHS, lr: float = LR,
          hidden: int | None = None, balance: bool = True,
          seed: int = 0) -> dict:
    """k-fold real-vs-synthetic accuracy on two sets of frozen embeddings.

    Synthetic is class 1, real class 0.  With `balance` the larger set is
    subsampled to the smaller one so plain accuracy shares the 0.5 chance level
    with the balanced accuracy; both are reported either way.  `l2` defaults per
    classifier (`L2` / `L2_MLP`) and is echoed back for reproducibility.
    """
    if l2 is None:
        l2 = L2 if hidden is None else L2_MLP
    x_syn = np.atleast_2d(np.asarray(x_syn, dtype=np.float64))
    x_real = np.atleast_2d(np.asarray(x_real, dtype=np.float64))
    if x_syn.shape[1] != x_real.shape[1]:
        raise ValueError(f"embedding size mismatch: "
                         f"{x_syn.shape[1]} vs {x_real.shape[1]}")
    if balance:
        rng = np.random.default_rng(seed)
        n = min(len(x_syn), len(x_real))
        x_syn = x_syn[rng.choice(len(x_syn), n, replace=False)]
        x_real = x_real[rng.choice(len(x_real), n, replace=False)]

    x = np.vstack([x_syn, x_real])
    y = np.concatenate([np.ones(len(x_syn)), np.zeros(len(x_real))])
    if min(len(x_syn), len(x_real)) < folds:
        raise ValueError(f"need >= {folds} clips per class for {folds}-fold CV")

    accs, baccs = [], []
    for k, test_idx in enumerate(_stratified_folds(y, folds, seed)):
        mask = np.ones(len(y), dtype=bool)
        mask[test_idx] = False
        pred = _fit_predict(x[mask], y[mask], x[test_idx],
                            l2, epochs, lr, hidden, seed + k)
        accs.append(float((pred == y[test_idx]).mean()))
        baccs.append(_balanced_accuracy(y[test_idx], pred))
    accs, baccs = np.asarray(accs), np.asarray(baccs)
    mean_bacc = float(baccs.mean())
    return {
        "accuracy": round(float(accs.mean()), 4),
        "accuracy_std": round(float(accs.std()), 4),
        "balanced_accuracy": round(mean_bacc, 4),
        "balanced_accuracy_std": round(float(baccs.std()), 4),
        "fold_accuracies": [round(float(a), 4) for a in accs],
        "n_synthetic": int(len(x_syn)),
        "n_real": int(len(x_real)),
        "dim": int(x.shape[1]),
        "folds": folds,
        "classifier": "logreg" if hidden is None else f"mlp{hidden}",
        "l2": l2,
        "passes_gate": bool(mean_bacc <= GATE_ACCEPT),
        "verdict": ("good" if mean_bacc <= GATE_GOOD else
                    "acceptable" if mean_bacc <= GATE_ACCEPT else "separable"),
    }


def null_control(x: np.ndarray, seed: int = 0, folds: int = FOLDS,
                 **probe_kwargs) -> dict | None:
    """Probe one set against a random half-split of itself.

    The empirical chance floor for this feature size and set size: whatever
    accuracy this reaches, the real-vs-synthetic number has to beat it before
    it means anything.  None when the set is too small to split k-fold.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    half = len(x) // 2
    if half < folds:
        return None
    idx = np.random.default_rng(seed).permutation(len(x))
    return probe(x[idx[:half]], x[idx[half:2 * half]], folds=folds, seed=seed,
                 **probe_kwargs)


def probe_dirs(syn_dir, real_dir, layer: int = WAVLM_LAYER, device=None,
               null: bool = True, **probe_kwargs) -> dict:
    """Embed both directories with one WavLM layer, then run the probe."""
    embedder = wavlm_embedder(layer=layer, device=device)
    t0 = time.time()
    _, syn = embed_clips(syn_dir, embedder)
    _, real = embed_clips(real_dir, embedder)
    out = probe(syn, real, **probe_kwargs)
    if null:
        floor = null_control(real, **probe_kwargs)
        out["null_balanced_accuracy"] = (floor or {}).get("balanced_accuracy")
    out.update({"layer": layer, "synthetic_dir": str(syn_dir),
                "real_dir": str(real_dir), "seconds": round(time.time() - t0, 1)})
    return out


def layer_sweep(syn_dir, real_dir, layers: Sequence[int] = SWEEP_LAYERS,
                device=None, null: bool = True, **probe_kwargs) -> dict:
    """Probe accuracy per WavLM layer, one forward pass per clip for all layers.

    The diagnostic of 05 §2: the layers where accuracy peaks are the ones whose
    features carry the synthetic-vs-real difference -- read against that layer's
    own null floor, since the floor drifts a little from layer to layer.
    """
    layers = tuple(layers)
    embedder = wavlm_embedder(layer=layers, device=device)
    t0 = time.time()
    _, syn = embed_clips(syn_dir, embedder)        # (N, L, D)
    _, real = embed_clips(real_dir, embedder)
    per_layer = {}
    for i, layer in enumerate(layers):
        entry = probe(syn[:, i], real[:, i], **probe_kwargs)
        if null:
            floor = null_control(real[:, i], **probe_kwargs)
            entry["null_balanced_accuracy"] = (floor or {}).get("balanced_accuracy")
        per_layer[str(layer)] = entry
    best = max(per_layer.items(), key=lambda kv: kv[1]["balanced_accuracy"])
    return {
        "synthetic_dir": str(syn_dir), "real_dir": str(real_dir),
        "layers": list(layers), "per_layer": per_layer,
        "best_layer": int(best[0]),
        "best_balanced_accuracy": best[1]["balanced_accuracy"],
        "seconds": round(time.time() - t0, 1),
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Tier 2 real-vs-synthetic probe")
    ap.add_argument("syn_dir", help="synthetic wav directory")
    ap.add_argument("real_dir", help="real reference wav directory")
    ap.add_argument("--layer", type=int, default=WAVLM_LAYER,
                    help="WavLM hidden_states index (default %(default)s)")
    ap.add_argument("--sweep", action="store_true",
                    help="accuracy for every layer instead of just --layer")
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument("--hidden", type=int, help="use an MLP with this hidden size")
    ap.add_argument("--no-null", action="store_true",
                    help="skip the real-vs-real chance-floor control")
    ap.add_argument("--device", help="torch device (default: cuda, else mps, else cpu)")
    ap.add_argument("--out", help="write the full JSON report here")
    args = ap.parse_args(argv)

    kwargs = {"folds": args.folds, "hidden": args.hidden}
    if args.sweep:
        res = layer_sweep(args.syn_dir, args.real_dir, device=args.device,
                          null=not args.no_null, **kwargs)
        for layer in res["layers"]:
            r = res["per_layer"][str(layer)]
            print(f"  layer {layer:>2}  balanced_acc={r['balanced_accuracy']:.3f}"
                  f" +/- {r['balanced_accuracy_std']:.3f}  "
                  f"null={_fmt(r.get('null_balanced_accuracy'))}  {r['verdict']}")
        print(f"worst case: layer {res['best_layer']} at "
              f"{res['best_balanced_accuracy']:.3f} [{res['seconds']}s]")
    else:
        res = probe_dirs(args.syn_dir, args.real_dir, layer=args.layer,
                         device=args.device, null=not args.no_null, **kwargs)
        print(f"layer {res['layer']}  balanced_acc={res['balanced_accuracy']:.3f}"
              f" +/- {res['balanced_accuracy_std']:.3f}  ({res['verdict']}, "
              f"gate <= {GATE_ACCEPT})  n={res['n_synthetic']}/{res['n_real']}"
              f"  [{res['seconds']}s]")
        print(f"  folds: {res['fold_accuracies']}")
        print(f"  null floor (real split in half): "
              f"{_fmt(res.get('null_balanced_accuracy'))}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")
    return res


if __name__ == "__main__":
    main()
