"""One-page HTML evaluation report (05-evaluation-plan.md §2, listening protocol).

Overlay histograms of the Tier 1 statistics plus the mean LTAS curve, rendered
with matplotlib and embedded as base64 PNGs, next to `<audio>` players for the
fixed audition list (20 samples per generator version, rendered against real
clips). matplotlib is imported lazily so the rest of `atcgen.eval` works
without the `[eval]` extra installed.
"""

import base64
import html
import io
import os
from pathlib import Path

from .channel_stats import SCALAR_KEYS

_CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem auto;
       max-width: 1000px; color: #222; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
table { border-collapse: collapse; font-size: 0.85rem; }
th, td { border: 1px solid #ddd; padding: 3px 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.bad { background: #fde2e2; } .good { background: #e6f5e6; }
img { max-width: 100%; } audio { width: 260px; }
.aud { display: inline-block; margin: 0 1rem 1rem 0; font-size: 0.8rem; }
"""


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _hist_figure(syn: dict, real: dict | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for ax, key in zip(axes.ravel(), SCALAR_KEYS):
        s = [c[key] for c in syn["clips"]]
        ax.hist(s, bins=25, alpha=0.6, density=True, label="synthetic", color="#3572b0")
        if real:
            ax.hist([c[key] for c in real["clips"]], bins=25, alpha=0.5,
                    density=True, label="real", color="#d1701c")
        ax.set_title(key, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(SCALAR_KEYS):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def _ltas_figure(syn: dict, real: dict | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogx(syn["ltas_hz"], syn["ltas_db_mean"], label="synthetic", color="#3572b0")
    if real:
        ax.semilogx(real["ltas_hz"], real["ltas_db_mean"], label="real", color="#d1701c")
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB rel. clip total")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _summary_table(syn: dict, real: dict | None, comparison: dict | None) -> str:
    head = ["statistic", "syn p10", "syn p50", "syn p90"]
    if real:
        head += ["real p10", "real p50", "real p90"]
    if comparison:
        head += ["Wasserstein", "median in p10-p90"]
    rows = ["<tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for key in SCALAR_KEYS:
        s = syn["summary"][key]
        cells = [key] + [f"{s[f'p{q}']:.2f}" for q in (10, 50, 90)]
        if real:
            r = real["summary"][key]
            cells += [f"{r[f'p{q}']:.2f}" for q in (10, 50, 90)]
        cls = ""
        if comparison:
            c = comparison["stats"][key]
            cells += [f"{c['wasserstein']:.3f}", "yes" if c["median_in_range"] else "NO"]
            cls = ' class="good"' if c["median_in_range"] else ' class="bad"'
        rows.append(f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _audition_html(audition, out_dir: Path) -> str:
    blocks = []
    for item in audition:
        label, path = item if isinstance(item, (tuple, list)) else (item.get("label", ""), item["audio"])
        rel = os.path.relpath(Path(path).resolve(), out_dir.resolve())
        blocks.append(
            f'<div class="aud">{html.escape(str(label))}<br>'
            f'<audio controls preload="none" src="{html.escape(rel)}"></audio></div>'
        )
    return "".join(blocks)


def build_report(out_path: str | Path, synthetic: dict, real: dict | None = None,
                 comparison: dict | None = None, audition=(),
                 title: str = "Synthetic ATC audio — evaluation report",
                 qc_summary: dict | None = None) -> Path:
    """Write the one-page HTML report. `synthetic`/`real` are `compute_stats`
    outputs, `comparison` a `compare` output, `audition` a list of
    (label, wav_path) pairs (paths are linked relative to the report)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    parts = [f"<h1>{html.escape(title)}</h1>",
             f"<p>{synthetic['n']} synthetic clips"
             + (f" vs {real['n']} real clips" if real else "") + "</p>"]
    if qc_summary:
        parts.append("<h2>Tier 0 QC</h2><table>"
                     + f"<tr><th>total</th><td>{qc_summary.get('total')}</td></tr>"
                     + f"<tr><th>discard rate</th><td>{qc_summary.get('discard_rate')}</td></tr>"
                     + "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>"
                               for k, v in qc_summary.get("reasons", {}).items())
                     + "</table>")
    parts.append("<h2>Tier 1 statistics</h2>"
                 + _summary_table(synthetic, real, comparison))
    if comparison:
        parts.append(f"<p>LTAS L1 distance: {comparison['ltas_l1_db']:.2f} dB &middot; "
                     f"all medians in range: "
                     f"<b>{'yes' if comparison['all_medians_in_range'] else 'NO'}</b></p>")
    parts.append(f'<h2>Distributions</h2><img src="data:image/png;base64,'
                 f'{_png(_hist_figure(synthetic, real))}">')
    parts.append(f'<h2>LTAS</h2><img src="data:image/png;base64,'
                 f'{_png(_ltas_figure(synthetic, real))}">')
    if audition:
        parts.append("<h2>Audition</h2>" + _audition_html(audition, out.parent))

    out.write_text(f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>"
                   f"<style>{_CSS}</style>" + "".join(parts))
    return out
