"""Tier 1 embedding distances (see docs/plans/05-evaluation-plan.md §2).

Two embedding families, per 00-research-findings.md §5 (never VGGish):

  wavlm  time-averaged hidden states of `microsoft/wavlm-base-plus`.  The
         layer is a parameter because WavLM layers encode channel information
         (2501.05310) and which layer separates the domains is itself the
         diagnostic; the default is a mid layer.  Raw (unnormalized) features.
  clap   `laion/clap-htsat-unfused` audio tower via transformers' own CLAP
         implementation -- no `laion-clap` package needed.  Wants 48 kHz, so
         clips are resampled; features are L2-normalized as CLAP's joint
         space intends.

Distances over two sets of embeddings:

  kid       polynomial-kernel MMD^2, unbiased, averaged over random subsets.
            Primary metric: unlike Frechet it is unbiased, so it stays near
            zero for same-distribution sets at our n~100 reference size.
  frechet   FAD-style Gaussian Frechet distance.  Secondary and, with n well
            below the 768-d feature size, badly biased upward -- `compare`
            flags that case rather than hiding it.

Embedders are `(wav, sr) -> vector` callables loaded lazily, so tests can
inject a fake and importing this module never touches the network.

CLI: python -m atcgen.eval.embed_dist <syn_dir> <real_dir> [--out d.json]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .channel_stats import _iter_clips

WAVLM_MODEL = "microsoft/wavlm-base-plus"
CLAP_MODEL = "laion/clap-htsat-unfused"
WAVLM_LAYER = 7             # mid transformer layer (hidden_states index)
WAVLM_SR = 16000
CLAP_SR = 48000
MAX_SECONDS = 10.0          # bound the forward pass; ATC clips are ~5 s
KID_SUBSETS = 100
KID_SUBSET_SIZE = 50
KID_DEGREE = 3

Embedder = Callable[[np.ndarray, int], np.ndarray]


# --- embedders ---------------------------------------------------------------

def _torch_device(device=None):
    import torch

    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _prep(wav: np.ndarray, sr: int, target_sr: int, max_seconds: float) -> np.ndarray:
    """Mono float32 at `target_sr`, truncated to `max_seconds`."""
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
    limit = int(target_sr * max_seconds)
    return np.ascontiguousarray(x[:limit])


def wavlm_embedder(layer: int | Sequence[int] = WAVLM_LAYER,
                   model_name: str = WAVLM_MODEL, device=None,
                   max_seconds: float = MAX_SECONDS) -> Embedder:
    """Time-averaged WavLM hidden states, model loaded on first call.

    `layer` indexes `hidden_states`: 0 is the CNN feature projection, 1-12 the
    transformer layers.  Passing a sequence returns an (n_layers, 768) stack
    from a single forward pass, which is what the per-layer probe sweep uses.
    """
    layers = (layer,) if isinstance(layer, int) else tuple(layer)
    state: dict = {}

    def embed(wav: np.ndarray, sr: int) -> np.ndarray:
        import torch

        if not state:
            from transformers import AutoFeatureExtractor, WavLMModel

            dev = _torch_device(device)
            state["dev"] = dev
            state["fe"] = AutoFeatureExtractor.from_pretrained(model_name)
            state["model"] = WavLMModel.from_pretrained(model_name).to(dev).eval()
        x = _prep(wav, sr, WAVLM_SR, max_seconds)
        inputs = state["fe"](x, sampling_rate=WAVLM_SR, return_tensors="pt")
        inputs = {k: v.to(state["dev"]) for k, v in inputs.items()}
        with torch.no_grad():
            out = state["model"](**inputs, output_hidden_states=True)
        stack = np.stack([out.hidden_states[i][0].mean(0).float().cpu().numpy()
                          for i in layers])
        return stack[0] if isinstance(layer, int) else stack

    return embed


def clap_embedder(model_name: str = CLAP_MODEL, device=None,
                  max_seconds: float = MAX_SECONDS,
                  normalize: bool = True) -> Embedder:
    """CLAP audio-tower embedding (512-d), model loaded on first call."""
    state: dict = {}

    def embed(wav: np.ndarray, sr: int) -> np.ndarray:
        import torch

        if not state:
            from transformers import ClapModel, ClapProcessor

            dev = _torch_device(device)
            state["dev"] = dev
            state["proc"] = ClapProcessor.from_pretrained(model_name)
            state["model"] = ClapModel.from_pretrained(model_name).to(dev).eval()
        x = _prep(wav, sr, CLAP_SR, max_seconds)
        inputs = state["proc"](audio=x, sampling_rate=CLAP_SR,
                               return_tensors="pt")
        inputs = {k: v.to(state["dev"]) for k, v in inputs.items()}
        with torch.no_grad():
            out = state["model"].get_audio_features(**inputs)
        # transformers >=5 returns BaseModelOutputWithPooling (the projected
        # 512-d embedding is `pooler_output`); older versions return it bare.
        feat = getattr(out, "pooler_output", out)
        v = feat[0].float().cpu().numpy()
        return v / (np.linalg.norm(v) + 1e-12) if normalize else v

    return embed


EMBEDDERS: dict[str, Callable[..., Embedder]] = {
    "wavlm": wavlm_embedder,
    "clap": clap_embedder,
}


def embed_clips(source, embedder: Embedder, sr: int | None = None
                ) -> tuple[list[str], np.ndarray]:
    """Embed a wav directory, a list of paths, or a list of arrays.

    Returns the clip names and a float64 array stacked on axis 0 -- (N, D) for
    a single-vector embedder, (N, L, D) for a multi-layer one.
    """
    names, vecs = [], []
    for name, wav, file_sr in _iter_clips(source, sr):
        if len(wav) == 0:
            continue
        names.append(name)
        vecs.append(np.asarray(embedder(wav, file_sr), dtype=np.float64))
    if not vecs:
        raise ValueError(f"no clips found in {source}")
    return names, np.stack(vecs)


# --- distances ---------------------------------------------------------------

def _poly_kernel(a: np.ndarray, b: np.ndarray, degree: int, gamma: float | None,
                 coef0: float) -> np.ndarray:
    g = 1.0 / a.shape[1] if gamma is None else gamma
    return (g * (a @ b.T) + coef0) ** degree


def _mmd2_unbiased(x: np.ndarray, y: np.ndarray, degree: int,
                   gamma: float | None, coef0: float) -> float:
    """Unbiased MMD^2 estimate (diagonal terms dropped, so it can go slightly
    negative for same-distribution sets -- that is the point of unbiasedness)."""
    m, n = len(x), len(y)
    kxx = _poly_kernel(x, x, degree, gamma, coef0)
    kyy = _poly_kernel(y, y, degree, gamma, coef0)
    kxy = _poly_kernel(x, y, degree, gamma, coef0)
    xx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    yy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return float(xx + yy - 2.0 * kxy.mean())


def kid(x: np.ndarray, y: np.ndarray, subsets: int = KID_SUBSETS,
        subset_size: int | None = KID_SUBSET_SIZE, degree: int = KID_DEGREE,
        gamma: float | None = None, coef0: float = 1.0, seed: int = 0) -> dict:
    """Kernel Inception/Audio Distance: subset-averaged unbiased polynomial MMD^2.

    `subset_size` is clamped to the smaller set (and to >= 2); with subsets
    drawn without replacement the mean is comparable across set sizes, which
    the Frechet distance is not.  Returns mean, std and the settings used.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"embedding size mismatch: {x.shape[1]} vs {y.shape[1]}")
    size = min(len(x), len(y), subset_size or max(len(x), len(y)))
    if size < 2:
        raise ValueError("need at least 2 clips per set")
    rng = np.random.default_rng(seed)
    vals = [_mmd2_unbiased(x[rng.choice(len(x), size, replace=False)],
                           y[rng.choice(len(y), size, replace=False)],
                           degree, gamma, coef0) for _ in range(subsets)]
    vals = np.asarray(vals)
    return {
        "kid": float(vals.mean()),
        "kid_std": float(vals.std()),
        "subsets": subsets,
        "subset_size": size,
        "degree": degree,
    }


def frechet(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """FAD-style Gaussian Frechet distance between two embedding sets.

    Strongly biased upward when the set size is not well above the feature
    dimension (the covariance estimate is then rank-deficient); `compare`
    reports that condition alongside the number.
    """
    from scipy import linalg

    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    mu_x, mu_y = x.mean(0), y.mean(0)
    cov_x = np.cov(x, rowvar=False) + eps * np.eye(x.shape[1])
    cov_y = np.cov(y, rowvar=False) + eps * np.eye(y.shape[1])
    covmean = linalg.sqrtm(cov_x @ cov_y)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_x - mu_y
    return float(diff @ diff + np.trace(cov_x) + np.trace(cov_y)
                 - 2.0 * np.trace(covmean))


def compare(syn: np.ndarray, real: np.ndarray, **kid_kwargs) -> dict:
    """KID (primary) + Frechet (secondary) between two embedding sets."""
    out = dict(kid(syn, real, **kid_kwargs))
    out.update({
        "frechet": round(frechet(syn, real), 4),
        "n_synthetic": int(len(syn)),
        "n_real": int(len(real)),
        "dim": int(np.atleast_2d(syn).shape[1]),
        # FAD's small-n bias (05 §1): flag when n is not well above the dim.
        "frechet_reliable": bool(min(len(syn), len(real))
                                 > 2 * np.atleast_2d(syn).shape[1]),
    })
    out["kid"] = round(out["kid"], 6)
    out["kid_std"] = round(out["kid_std"], 6)
    return out


def compare_dirs(syn_dir, real_dir, families: Sequence[str] = ("wavlm", "clap"),
                 wavlm_layer: int = WAVLM_LAYER, device=None,
                 **kid_kwargs) -> dict:
    """Embed both directories with each family and report the distances.

    JSON-serializable; embeddings themselves are not returned (use
    `embed_clips` when you want to keep them, e.g. for the Tier 2 probe).
    """
    results: dict = {"synthetic_dir": str(syn_dir), "real_dir": str(real_dir),
                     "families": {}}
    for family in families:
        if family not in EMBEDDERS:
            raise ValueError(f"unknown embedding family {family!r}")
        kwargs = {"device": device}
        if family == "wavlm":
            kwargs["layer"] = wavlm_layer
        embedder = EMBEDDERS[family](**kwargs)
        t0 = time.time()
        _, syn = embed_clips(syn_dir, embedder)
        _, real = embed_clips(real_dir, embedder)
        entry = compare(syn, real, **kid_kwargs)
        entry["seconds"] = round(time.time() - t0, 1)
        if family == "wavlm":
            entry["layer"] = wavlm_layer
        results["families"][family] = entry
    return results


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Tier 1 embedding distances")
    ap.add_argument("syn_dir", help="synthetic wav directory")
    ap.add_argument("real_dir", help="real reference wav directory")
    ap.add_argument("--families", default="wavlm,clap",
                    help="comma-separated: wavlm,clap")
    ap.add_argument("--layer", type=int, default=WAVLM_LAYER,
                    help="WavLM hidden_states index (default %(default)s)")
    ap.add_argument("--subsets", type=int, default=KID_SUBSETS)
    ap.add_argument("--subset-size", type=int, default=KID_SUBSET_SIZE)
    ap.add_argument("--device", help="torch device (default: cuda, else mps, else cpu)")
    ap.add_argument("--out", help="write the full JSON report here")
    args = ap.parse_args(argv)

    res = compare_dirs(args.syn_dir, args.real_dir,
                       families=tuple(args.families.split(",")),
                       wavlm_layer=args.layer, device=args.device,
                       subsets=args.subsets, subset_size=args.subset_size)
    for family, r in res["families"].items():
        print(f"{family}: KID={r['kid']:+.6f} +/- {r['kid_std']:.6f}  "
              f"Frechet={r['frechet']:.3f}"
              f"{'' if r['frechet_reliable'] else ' (small-n biased)'}  "
              f"n={r['n_synthetic']}/{r['n_real']} d={r['dim']}  "
              f"[{r['seconds']}s]")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")
    return res


if __name__ == "__main__":
    main()
