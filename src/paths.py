"""Repository paths shared by every script.

Figures always go to paper/figures/, the folder synced with Overleaf, so a re-run of any
plotting script updates the paper on the next push.

    from paths import save_figure
    save_figure(fig, "nu_convergence")          # -> paper/figures/nu_convergence.pdf (+ .png preview)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG_DIR = ROOT / "paper" / "figures"


def figure_path(name, ext=".pdf"):
    """paper/figures/<stem of name><ext>; any folder or extension in `name` is dropped."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR / (Path(str(name)).stem + ext)


def save_figure(fig, name, formats=(".pdf", ".png"), **savefig_kwargs):
    """Save `fig` into paper/figures/ in each format and return the PDF path.

    The PDF is what LaTeX includes; the PNG is a local preview (*.png is git-ignored).
    """
    savefig_kwargs.setdefault("bbox_inches", "tight")
    savefig_kwargs.setdefault("dpi", 300)
    paths = [figure_path(name, ext) for ext in formats]
    for p in paths:
        fig.savefig(p, **savefig_kwargs)
    return paths[0]
