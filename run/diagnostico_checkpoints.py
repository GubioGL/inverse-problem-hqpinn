"""
Pós-processamento dos checkpoints dos seus scripts Inverser_*_pinn_paralelo.py.
Coloque este arquivo (e benchmark_viscosidade.py) na mesma pasta dos scripts.

Para cada configuração (modelo, T, N_col, largura) calcula:
  - NU realmente usado (avisa se classical e quantum usam NU diferentes);
  - viés = média(nu_hat - NU), desvio-padrão, MSE = viés^2 + var, IC 95% bootstrap do MSE,
    mediana do erro relativo e fração de runs que atingiram target_loss;
  - o piso de viés dos dados FD (nu_efetivo - NU) para comparação direta
    (zero para os runs v2, cujos dados vêm da solução exata);
  - com --denso: nu reajustado no modelo congelado usando 20 000 pontos
    (nu = sum(A B)/sum(B^2)), para separar a variância que vem de o refinamento
    final usar só os N_col últimos pontos;
  - com --denso: erro relativo L2 de u(.,T) contra a solução exata e o erro em nu
    "previsto" por esse erro via kappa (erro_rel(nu) ~ erro_rel(u) / kappa).

Uso:
  python diagnostico_checkpoints.py --classico data/classical_inverse \
         --quantico data/quantum_inverse_exact [--denso] --saida resumo.csv
(Funciona com checkpoints v1 e v2; testado em checkpoints v2 de runs curtos.)
"""

# --- repository layout: src/ and run/ go on the import path, and relative paths
# --- such as "data/..." are resolved from the repository root.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "src").is_dir() and (p / "run").is_dir())
for _p in (_ROOT / "run", _ROOT / "src", _ROOT / "src" / "quantum_hardware", _HERE):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
_os.chdir(_ROOT)
# ---------------------------------------------------------------------------------

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

from benchmark_viscosidade import cole_hopf, fit_nu, setup

_floor_cache, _kappa_cache = {}, {}


def data_floor(nu, T):
    """nu_efetivo(dados FD) - nu, i.e. o viés que qualquer estimador herda dos dados."""
    if (nu, T) not in _floor_cache:
        s = setup(nu, T)
        _floor_cache[(nu, T)] = fit_nu(s["xd"], s["T"], s["u_fd"], nu) - nu
        _kappa_cache[(nu, T)] = nu * np.linalg.norm(s["S"]) / np.linalg.norm(s["u_ex"])
    return _floor_cache[(nu, T)]


def kappa(nu, T):
    """Condicionamento kappa = nu ||du/dnu|| / ||u|| (dados em t = T)."""
    if (nu, T) not in _kappa_cache:
        s = setup(nu, T)
        _kappa_cache[(nu, T)] = nu * np.linalg.norm(s["S"]) / np.linalg.norm(s["u_ex"])
    return _kappa_cache[(nu, T)]


def load_records(folder, kind, dense):
    import torch
    if kind == "classico":
        import Inverser_classical_pinn_paralelo as mod
    else:
        import Inverser_quantum_exact_pinn_paralelo as mod
    recs = []
    for path in sorted(glob.glob(os.path.join(folder, "T_*", "model_*.pt"))):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        tp = ck["training_params"]
        width = tp.get("INTRA_NEURON", tp.get("N_QUBITS"))
        rec = dict(kind=kind, T=float(tp["TFIN"]), nu=float(tp["NU"]), ncol=int(tp["N_COLLOCATION"]),
                   width=int(width), nu_hat=float(tp["NU_LEARNED"]),
                   exact_data=str(tp.get("DATA_SOURCE", "fd")).startswith("exact"),
                   reached=tp.get("TARGET_REACHED"))   # None in v2 (no target_loss)
        if dense:
            if kind == "classico":
                model = mod.build_model(rec["width"], rec["T"], 0.1)
                model.load_state_dict(ck["state_dict"])
            else:
                model = mod.QuantumExactInversePINN.load(path)
            model.eval()
            g = torch.Generator().manual_seed(123)
            n = 20000
            x = torch.rand((n, 1), generator=g) * mod.L
            t = mod.HT + torch.rand((n, 1), generator=g) * (rec["T"] - 2 * mod.HT)
            with torch.no_grad():
                a, b = mod.residual_terms(model(mod.stencil_inputs(x, t, mod.L, mod.HX, mod.HT)),
                                          n, mod.HX, mod.HT)
                rec["nu_dense"] = float((a * b).sum() / (b * b).sum())
                xs = np.linspace(0, 1, 200, endpoint=False)
                u_p = mod.predict(model, xs, rec["T"]).ravel()
            u_e = cole_hopf(xs, rec["T"], rec["nu"])
            rec["erro_u_T"] = float(np.linalg.norm(u_p - u_e) / np.linalg.norm(u_e))
        recs.append(rec)
    return recs


def summarize(recs, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for r in recs:
        groups[(r["kind"], r["T"], r["ncol"], r["width"])].append(r)
    nus_by_kind = defaultdict(set)
    rows = []
    for key in sorted(groups):
        g = groups[key]
        nu_set = {r["nu"] for r in g}
        nus_by_kind[key[0]] |= nu_set
        nu = g[0]["nu"]
        err = np.array([r["nu_hat"] - r["nu"] for r in g])
        boot = [np.mean(rng.choice(err, err.size) ** 2) for _ in range(n_boot)]
        row = dict(modelo=key[0], T=key[1], N_col=key[2], largura=key[3], n_runs=err.size, NU=nu,
                   vies=err.mean(), desvio=err.std(ddof=1) if err.size > 1 else np.nan,
                   MSE=np.mean(err ** 2), MSE_IC95_inf=np.percentile(boot, 2.5),
                   MSE_IC95_sup=np.percentile(boot, 97.5),
                   mediana_erro_rel=np.median(np.abs(err)) / nu,
                   frac_target=(np.mean([bool(r["reached"]) for r in g])
                                if g[0]["reached"] is not None else np.nan),
                   # dados exatos (v2): sem viscosidade numérica -> piso ~0
                   piso_vies_dados=0.0 if g[0]["exact_data"] else data_floor(nu, key[1]))
        if "nu_dense" in g[0]:
            ed = np.array([r["nu_dense"] - r["nu"] for r in g])
            eu = np.array([r["erro_u_T"] for r in g])
            row.update(MSE_denso=np.mean(ed ** 2), vies_denso=ed.mean(),
                       erro_u_T_mediano=np.median(eu),
                       erro_nu_previsto_mediano=np.median(eu) / kappa(nu, key[1]) * nu)
        rows.append(row)
    if len(nus_by_kind) > 1 and len(set.union(*nus_by_kind.values())) > 1:
        print("ATENÇÃO: NU diferente entre modelos:", dict(nus_by_kind),
              "-> MSE(nu) absoluto não é diretamente comparável.")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classico")
    ap.add_argument("--quantico")
    ap.add_argument("--denso", action="store_true")
    ap.add_argument("--saida", default="resumo_nu.csv")
    args = ap.parse_args()
    recs = []
    if args.classico:
        recs += load_records(args.classico, "classico", args.denso)
    if args.quantico:
        recs += load_records(args.quantico, "quantico", args.denso)
    rows = summarize(recs)
    with open(args.saida, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"{len(rows)} configurações salvas em {args.saida}")
