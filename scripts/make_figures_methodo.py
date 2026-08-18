#!/usr/bin/env python
"""
Figures complementaires pour les sections 3.2 et 3.3.

    fig8_variables.png   comptabilite des variables binaires        (3.2)
    fig9_composition.png composition du cout                         (3.2)
    fig10_seuil.png      seuil de marge de capacite                  (3.3)

La figure 10 lit results/seuil_marge.json, produit par
scripts/seuil_marge_capacite.py. Les deux autres sont autonomes, a une reserve
pres signalee dans fig9.

Usage :
    python scripts/make_figures_methodo.py --outdir figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_EXACT = "#2E5266"
C_HYB = "#C06014"
C_TRANSPORT = "#9BB1BF"
C_HUBS = "#6E8898"
C_VEH = "#D9A15B"
C_GRID = "#D8D4CC"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.6,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ecrit {path}")


# ------------------------------------------------------------------ 3.2 (a)
def fig8_variables(outdir):
    """Comptabilite des variables, du produit cartesien au QUBO."""
    etapes = [
        ("Theoretical triples\n30 × 3 × 16", 1440, "#BFBAB0"),
        ("Admissible after\ntime-window filter", 848, C_EXACT),
        ("+ 3 hub\n+ 16 vehicle\n= MILP model", 867, C_EXACT),
        ("+ 81 slack\n= joint QUBO", 948, C_HYB),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = range(len(etapes))
    ax.bar(xs, [e[1] for e in etapes], color=[e[2] for e in etapes],
           width=0.58, zorder=3)
    for x, (_, v, _) in zip(xs, etapes):
        ax.text(x, v + 26, f"{v:,}", ha="center", fontsize=9, zorder=4)

    # variation entre les deux premieres barres
    ax.annotate("", xy=(0.72, 848), xytext=(0.72, 1440),
                arrowprops=dict(arrowstyle="<|-", color="#888888", lw=1.0))
    ax.text(0.79, 1150, "−592\n(41 %)", fontsize=7.5, color="#666666",
            va="center", linespacing=1.3)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([e[0] for e in etapes], fontsize=8)
    ax.set_ylabel("Binary variables")
    ax.set_ylim(0, 1620)
    ax.set_axisbelow(True)
    ax.set_title("From the Cartesian product to the QUBO", fontsize=9, pad=8)
    save(fig, outdir, "fig8_variables.png")


# ------------------------------------------------------------------ 3.2 (b)
def fig9_composition(outdir, joint=None):
    """Composition du cout : trois termes, deux ou trois solutions.

    joint : dict {'hubs':…, 'vehicles':…, 'transport':…} pour l'optimum joint.
    Ces valeurs ne sont PAS deduisibles du total et doivent etre mesurees ;
    voir la note affichee si l'argument est omis.
    """
    solutions = []
    if joint:
        solutions.append(("Proven optimum\n8,532.96",
                          joint["hubs"], joint["vehicles"], joint["transport"]))
    solutions += [
        ("Exact, decomposed\nn = 4 · 9,493.73", 3300, 4400, 1793.73),
        ("Hybrid, typical run\nn = 4 · 9,847.37", 3300, 4800, 1747.37),
    ]

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    xs = range(len(solutions))
    hubs = [s[1] for s in solutions]
    vehs = [s[2] for s in solutions]
    trans = [s[3] for s in solutions]

    ax.bar(xs, hubs, color=C_HUBS, width=0.5, label="Hub opening", zorder=3)
    ax.bar(xs, vehs, bottom=hubs, color=C_VEH, width=0.5,
           label="Vehicle activation", zorder=3)
    ax.bar(xs, trans, bottom=[h + v for h, v in zip(hubs, vehs)],
           color=C_TRANSPORT, width=0.5, label="Transport", zorder=3)

    for x, s in zip(xs, solutions):
        tot = s[1] + s[2] + s[3]
        ax.text(x, tot + 140, f"{tot:,.0f}", ha="center", fontsize=8.5, zorder=4)
        fixe = (s[1] + s[2]) / tot
        ax.text(x, (s[1] + s[2]) / 2, f"{fixe:.0%}\nfixed", ha="center",
                va="center", fontsize=8, color="white", zorder=4, linespacing=1.3)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([s[0] for s in solutions], fontsize=8)
    ax.set_ylabel("Cost")
    ax.set_ylim(0, max(s[1] + s[2] + s[3] for s in solutions) * 1.16)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.set_axisbelow(True)
    ax.set_title("Fixed costs dominate the objective", fontsize=9, pad=8)
    save(fig, outdir, "fig9_composition.png")

    if not joint:
        print("    NOTE : la barre de l'optimum joint est absente, sa ventilation")
        print("    n'ayant pas ete mesuree. Commande :")
        print('      python -c "')
        print("      import sys, json; sys.path.insert(0, '.')")
        print("      from src.classical_model import build_classical_model")
        print("      from src.solve import solve_model")
        print("      from src.hybrid_pipeline import _pyomo_model_to_sample, "
              "recombine_solutions")
        print("      d = json.load(open('tests/fixtures/scaled_data_v2.json'))")
        print("      m = build_classical_model(d); solve_model(m, time_limit=300)")
        print("      s = _pyomo_model_to_sample(m, d)")
        print("      print(recombine_solutions([s], d)['cost_breakdown'])\"")


# ------------------------------------------------------------------ 3.3
def fig10_seuil(outdir, path="results/seuil_marge.json"):
    """Faisabilite en fonction de la marge de capacite."""
    if not os.path.exists(path):
        print(f"  Figure 10 ignoree : {path} absent")
        print("    lancez d'abord scripts/seuil_marge_capacite.py")
        return
    d = json.load(open(path))
    n = [l for l in d["niveaux"] if l["cout_moyen"] is not None
         or l["faisable"] == 0]
    if not n:
        print("  Figure 10 ignoree : aucun niveau exploitable")
        return

    mg = [l["marge_globale"] * 100 for l in n]
    mc = [l["marge_cluster_min"] * 100 for l in n]
    taux = [l["faisable"] / l["runs"] * 100 for l in n]
    lo = [l["ic_bas"] * 100 for l in n]
    hi = [l["ic_haut"] * 100 for l in n]

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.fill_between(mg, lo, hi, color=C_EXACT, alpha=0.15, zorder=2)
    ax.plot(mg, taux, "o-", color=C_EXACT, linewidth=1.6, markersize=5,
            zorder=3, label="Feasibility vs global margin")
    ax.plot(mc, taux, "s--", color=C_HYB, linewidth=1.4, markersize=4.5,
            zorder=3, label="Feasibility vs minimum cluster margin")

    ax.axvline(20, color="#999999", linestyle=":", linewidth=1.1, zorder=1)
    ax.annotate("20 % threshold\nas stated", xy=(20, 8), xytext=(24, 8),
                fontsize=7.5, color="#666666", va="center", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))

    ax.set_xlabel("Capacity margin (%)")
    ax.set_ylabel(f"Feasibility rate (%), {d['runs']} runs")
    ax.set_ylim(-5, 105)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=1)
    ax.set_axisbelow(True)
    ax.set_title("Feasibility against capacity margin\n"
                 f"(n = {d['n_clusters']}, fleet size held constant at 16 vehicles)",
                 fontsize=9, pad=8)
    save(fig, outdir, "fig10_seuil.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="figures")
    p.add_argument("--seuil", default="results/seuil_marge.json")
    p.add_argument("--joint", default=None,
                   help="JSON inline, ex. '{\"hubs\":2100,\"vehicles\":3900,"
                        "\"transport\":2532.96}'")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    print("Generation des figures methodologiques...")
    fig8_variables(a.outdir)
    fig9_composition(a.outdir, json.loads(a.joint) if a.joint else None)
    fig10_seuil(a.outdir, a.seuil)
    print("Termine.")


if __name__ == "__main__":
    main()