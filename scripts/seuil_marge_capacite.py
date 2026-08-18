#!/usr/bin/env python
"""
Etablit par la mesure le seuil de marge de capacite en dessous duquel la
methode hybride cesse de produire des solutions faisables.

Protocole. La capacite UNITAIRE des vehicules varie, leur NOMBRE reste fixe a
seize. C'est le seul moyen d'isoler la marge : reduire le nombre de vehicules
modifierait aussi la taille des sous-QUBO, et l'on ne saurait plus lequel des
deux facteurs explique un echec. Le ratio camionnette/camion est maintenu proche
de celui de l'instance de reference.

E_max est recalibre a chaque niveau, puisque le plancher d'emissions depend de
la capacite : une flotte plus contrainte force certains envois vers des vehicules
plus emetteurs.

AMBIGUITE LEVEE PAR CE SCRIPT. La section 3.3 enonce le seuil en marge GLOBALE,
la section 4.5 l'applique a la marge minimale d'un CLUSTER. Ce ne sont pas les
memes grandeurs : la flotte etant partitionnee, la marge d'un cluster est
generalement inferieure a la marge globale. Le script enregistre les deux, de
sorte que le seuil puisse etre enonce dans les termes voulus, et que l'on sache
si les deux lectures concordent.

Usage :
    python scripts/seuil_marge_capacite.py
    python scripts/seuil_marge_capacite.py --runs 15 --n-clusters 4
"""
import argparse
import contextlib
import copy
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, ".")

from pyomo.environ import Objective, minimize, value

from src.classical_model import build_classical_model
from src.solve import solve_model
from src.decomposition_agent import decompose_network
from src.qubo_model import build_qubo
from src.hybrid_pipeline import solve_hybrid

FIXTURE = "tests/fixtures/scaled_data_v2.json"
OUT = Path("results/seuil_marge.json")

# (capacite van, capacite truck) -> capacite totale 8*(v+t), poids total 183
NIVEAUX = [(11, 15), (12, 15), (12, 16), (13, 16), (13, 17), (14, 18), (15, 20)]

RATIOS = {"assignment": 20.4, "vehicle_activation": 6.3, "emissions": 1.3,
          "hub_activation": 1.2, "capacity": 1.0}
MARGE_EMAX = 1.25


def poids_penalite(data):
    mc = max(max(data["transport_cost"].values()),
             max(h["cost"] for h in data["hubs"].values()),
             max(v["fixed_cost"] for v in data["vehicles"].values()))
    return {k: mc * r for k, r in RATIOS.items()}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


@contextlib.contextmanager
def silence():
    """Supprime la sortie du solveur, y compris celle emise au niveau C.

    Sans cela le journal de Gurobi noie les lignes de resultat, et un simple
    redirect_stdout de Python ne suffit pas : la bibliotheque ecrit sur le
    descripteur de fichier 1 directement.
    """
    fd = sys.stdout.fileno()
    sauve = os.dup(fd)
    nul = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        os.dup2(nul, fd)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(sauve, fd)
        os.close(nul)
        os.close(sauve)


def plancher_emissions(sub):
    """Emissions minimales atteignables, plafond desactive.

    Renvoie None si le sous-probleme est infaisable. Ce cas se produit des que
    la capacite unitaire descend assez bas pour qu'un cluster ne puisse plus
    porter ses propres envois, alors meme que la marge GLOBALE reste positive.
    C'est un resultat et non une erreur : a ce niveau, la decomposition est
    structurellement impossible, sans qu'aucun recuit n'ait a etre lance.
    """
    d = dict(sub)
    d["E_max"] = 10 ** 9
    m = build_classical_model(d)
    m.del_component(m.OBJ)

    # Pas d'argument par defaut capturant d : Pyomo passerait un index
    # positionnel qui l'ecraserait silencieusement.
    def regle(mdl):
        return sum(d["emission"][f"{i}|{h}|{v}"] * mdl.x[i, h, v]
                   for (i, h, v) in mdl.VALID)

    m.OBJ = Objective(rule=regle, sense=minimize)
    with silence():
        solve_model(m, time_limit=120)
        try:
            return value(m.OBJ)
        except Exception:
            return None


def instance_a_capacite(base, cap_van, cap_truck):
    """Copie de l'instance avec les capacites unitaires modifiees."""
    d = copy.deepcopy(base)
    for v, vd in d["vehicles"].items():
        vd["capacity"] = cap_van if v.startswith("van") else cap_truck
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--sweeps", type=int, default=2000)
    p.add_argument("--fixture", default=FIXTURE)
    args = p.parse_args()

    base = json.load(open(args.fixture))
    poids_total = sum(s["weight"] for s in base["shipments"].values())
    poids_max = max(s["weight"] for s in base["shipments"].values())

    print(f"RES poids total {poids_total} kg, envoi le plus lourd {poids_max} kg")
    print(f"RES n_clusters = {args.n_clusters}, {args.runs} executions par niveau, "
          f"{args.sweeps} sweeps", flush=True)
    print("RES", flush=True)
    print("RES  van/truck | capacite | marge glob | marge clust min | bqm max | "
          "E_max  | optimum  | faisable | IC95      | cout moyen", flush=True)
    print("RES " + "-" * 118, flush=True)

    lignes = []
    for cap_van, cap_truck in NIVEAUX:
        if cap_van < poids_max:
            print(f"RES {cap_van:>4}/{cap_truck:<5} | capacite unitaire sous "
                  f"l'envoi le plus lourd -- niveau ignore", flush=True)
            continue

        data = instance_a_capacite(base, cap_van, cap_truck)
        capacite = sum(v["capacity"] for v in data["vehicles"].values())
        marge_glob = capacite / poids_total - 1

        if capacite <= poids_total:
            print(f"RES {cap_van:>4}/{cap_truck:<5} | {capacite:>8} | "
                  f"capacite totale sous le poids total -- infaisable", flush=True)
            continue

        # --- recalibration d'E_max sur la nouvelle flotte
        try:
            clusters = decompose_network(data, n_clusters=args.n_clusters)
        except Exception as e:
            print(f"RES {cap_van:>4}/{cap_truck:<5} | decomposition impossible : {e}",
                  flush=True)
            continue

        # --- marges et tailles, calculables meme si un cluster est infaisable
        marges_cl, tailles = [], []
        for sub in clusters.values():
            w = sum(s["weight"] for s in sub["shipments"].values())
            c = sum(v["capacity"] for v in sub["vehicles"].values())
            marges_cl.append(c / w - 1 if w else 99.0)
            with silence():
                bqm, _ = build_qubo(sub, poids_penalite(data))
            tailles.append(len(bqm.variables))
        marge_cl_min = min(marges_cl)

        # --- recalibration d'E_max sur la nouvelle flotte
        planchers = [plancher_emissions(s) for s in clusters.values()]
        if any(p is None for p in planchers):
            n_ko = sum(1 for p in planchers if p is None)
            print(f"RES {cap_van:>4}/{cap_truck:<5} | {capacite:>8} | "
                  f"{marge_glob:>9.1%} | {marge_cl_min:>14.1%} | {max(tailles):>7} | "
                  f"{'--':>6} | {'--':>8} | {0:>4}/{args.runs} | "
                  f"{'0%-0%':>9} | DECOMPOSITION INFAISABLE "
                  f"({n_ko} cluster(s) sans solution)", flush=True)
            lignes.append({
                "cap_van": cap_van, "cap_truck": cap_truck, "capacite": capacite,
                "marge_globale": round(marge_glob, 4),
                "marge_cluster_min": round(marge_cl_min, 4),
                "bqm_max": max(tailles), "E_max": None, "optimum_exact": None,
                "faisable": 0, "runs": args.runs, "ic_bas": 0.0, "ic_haut": 0.0,
                "cout_moyen": None, "ecart_type": None,
                "echec_structurel": True, "clusters_infaisables": n_ko,
            })
            continue

        data["E_max"] = round(sum(planchers) * MARGE_EMAX, 2)

        # --- optimum exact a ce niveau
        m = build_classical_model(data)
        with silence():
            solve_model(m, time_limit=600)
            try:
                optimum = value(m.OBJ)
            except Exception:
                optimum = None

        # --- campagne hybride
        P = poids_penalite(data)
        ok, couts = 0, []
        for _ in range(args.runs):
            with silence():
                r = solve_hybrid(data, n_clusters=args.n_clusters,
                                 penalty_weights=P, num_sweeps=args.sweeps)
            if r["feasible"]:
                ok += 1
                couts.append(r["cost"])
        lo, hi = wilson(ok, args.runs)
        moyen = st.mean(couts) if couts else None

        print(f"RES {cap_van:>4}/{cap_truck:<5} | {capacite:>8} | "
              f"{marge_glob:>9.1%} | {marge_cl_min:>14.1%} | {max(tailles):>7} | "
              f"{data['E_max']:>6.1f} | "
              f"{optimum if optimum else 0:>8.2f} | {ok:>4}/{args.runs} | "
              f"{lo:>4.0%}-{hi:<4.0%} | "
              f"{moyen if moyen else 0:>10.2f}", flush=True)

        lignes.append({
            "cap_van": cap_van, "cap_truck": cap_truck, "capacite": capacite,
            "marge_globale": round(marge_glob, 4),
            "marge_cluster_min": round(marge_cl_min, 4),
            "bqm_max": max(tailles),
            "E_max": data["E_max"],
            "optimum_exact": round(optimum, 2) if optimum else None,
            "faisable": ok, "runs": args.runs,
            "ic_bas": round(lo, 3), "ic_haut": round(hi, 3),
            "cout_moyen": round(moyen, 2) if moyen else None,
            "ecart_type": round(st.stdev(couts), 2) if len(couts) > 1 else None,
            "echec_structurel": False, "clusters_infaisables": 0,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_clusters": args.n_clusters, "runs": args.runs, "sweeps": args.sweeps,
        "poids_total": poids_total, "niveaux": lignes}, indent=2))
    print(f"RES", flush=True)
    print(f"RES -> {OUT}", flush=True)

    # --- lecture du seuil, dans les deux systemes de marge
    struct = [l for l in lignes if l["echec_structurel"]]
    if struct:
        pire = max(l["marge_globale"] for l in struct)
        print(f"RES echec STRUCTUREL jusqu'a {pire:.1%} de marge globale : "
              f"un cluster au moins ne peut porter ses envois, quel que soit "
              f"le solveur", flush=True)

    for cle, nom in [("marge_globale", "marge globale"),
                     ("marge_cluster_min", "marge minimale de cluster")]:
        succes = [l for l in lignes if l["faisable"] / l["runs"] >= 0.80]
        echecs = [l for l in lignes if l["faisable"] / l["runs"] < 0.80]
        if succes and echecs:
            bas = max(e[cle] for e in echecs)
            haut = min(s[cle] for s in succes)
            print(f"RES seuil en {nom} : entre {bas:.1%} et {haut:.1%} "
                  f"(critere : 80 % de faisabilite)", flush=True)
        else:
            print(f"RES seuil en {nom} : non encadre par les niveaux testes",
                  flush=True)


if __name__ == "__main__":
    main()