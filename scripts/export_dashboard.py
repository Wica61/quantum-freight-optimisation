#!/usr/bin/env python
"""
Construit results/dashboard.json, consomme par results/dashboard.html.

Deux sources distinctes :

  1. DETERMINISTE -- recalcule a chaque appel : l'instance, les partitions
     KMeans, la taille des sous-QUBO. Rapide, aucune mesure requise.

  2. MESURE -- lu depuis results/measures.json, que vos scripts alimentent
     via record(). Les sections absentes apparaissent comme "a mesurer"
     dans le tableau de bord : la page est sa propre liste de taches.

Usage :
    python scripts/export_dashboard.py
    open results/dashboard.html

Pour enregistrer une mesure depuis n'importe quel script :
    from scripts.export_dashboard import record
    record("baseline_jointe", {"faisable": 0, "runs": 5, "violations_min": 7})
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

FIXTURE = Path("tests/fixtures/scaled_data_v2.json")
MEASURES = Path("results/measures.json")
OUT = Path("results/dashboard.json")

# Reference : optimum joint prouve par Gurobi sur scaled_data_v2.
JOINT = 8532.96

RATIOS = {"assignment": 20.4, "vehicle_activation": 6.3, "emissions": 1.3,
          "hub_activation": 1.2, "capacity": 1.0}


# --------------------------------------------------------------------------
# Enregistrement des mesures
# --------------------------------------------------------------------------
def record(cle, valeur):
    """Ajoute ou remplace une section de mesures. Idempotent."""
    MEASURES.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if MEASURES.exists():
        data = json.loads(MEASURES.read_text())
    data[cle] = valeur
    MEASURES.write_text(json.dumps(data, indent=2))
    print(f"[measures] {cle} enregistre")


# --------------------------------------------------------------------------
# Partie deterministe
# --------------------------------------------------------------------------
def poids_penalite(data):
    mc = max(max(data["transport_cost"].values()),
             max(h["cost"] for h in data["hubs"].values()),
             max(v["fixed_cost"] for v in data["vehicles"].values()))
    return {k: mc * r for k, r in RATIOS.items()}, mc


def instance(data):
    S, V = data["shipments"], data["vehicles"]
    poids = sum(s["weight"] for s in S.values())
    capacite = sum(v["capacity"] for v in V.values())
    return {
        "n_envois": len(S),
        "n_hubs": len(data["hubs"]),
        "n_vehicules": len(V),
        "poids_total": poids,
        "capacite_totale": capacite,
        "marge_capacite": round(capacite / poids - 1, 4),
        "E_max": data["E_max"],
        "n_variables_x": sum(len(c) for c in data["valid_combinations"].values()),
        "n_variables_total": (sum(len(c) for c in data["valid_combinations"].values())
                              + len(data["hubs"]) + len(V)),
        "optimum_joint": JOINT,
        "hubs": {h: {"x": hd["x"], "y": hd["y"], "cost": hd["cost"]}
                 for h, hd in data["hubs"].items()},
        "envois": [{"id": i, "x": round(s["x"], 2), "y": round(s["y"], 2),
                    "w": s["weight"], "e": s["e"], "l": s["l"],
                    "n_combos": len(data["valid_combinations"][i])}
                   for i, s in S.items()],
    }


def partitions(data, valeurs_n):
    """Partition KMeans et taille reelle du QUBO pour chaque granularite.

    La taille du BQM inclut les variables d'ecart generees par les contraintes
    d'inegalite (capacite, emissions) -- elles representent 13 a 42 % du total
    et doivent apparaitre, sinon les chiffres du tableau de bord ne
    correspondent pas a ceux des logs.
    """
    from src.decomposition_agent import decompose_network
    from src.qubo_model import build_qubo

    poids, _ = poids_penalite(data)
    ids = list(data["shipments"])
    out = {}

    for n in valeurs_n:
        clusters = decompose_network(data, n_clusters=n)
        labels = {}
        details = []
        for idx, (cid, sub) in enumerate(sorted(clusters.items(),
                                                key=lambda kv: str(kv[0]))):
            for i in sub["shipments"]:
                labels[i] = idx
            bqm, _ = build_qubo(sub, poids)
            w = sum(s["weight"] for s in sub["shipments"].values())
            cap = sum(v["capacity"] for v in sub["vehicles"].values())
            n_x = sum(len(c) for c in sub["valid_combinations"].values())
            details.append({
                "cluster": idx,
                "n_envois": len(sub["shipments"]),
                "n_vehicules": len(sub["vehicles"]),
                "poids": w,
                "capacite": cap,
                "marge": round(cap / w - 1, 4) if w else None,
                "n_variables_x": n_x,
                "n_variables_bqm": len(bqm.variables),
                "n_slack": len(bqm.variables) - n_x - len(sub["hubs"]) - len(sub["vehicles"]),
                "E_alloue": sub.get("E_max"),
                "plancher": round(sub["_floor"], 2) if "_floor" in sub else None,
            })
        out[str(n)] = {
            "labels": [labels.get(i, -1) for i in ids],
            "clusters": details,
            "bqm_max": max(d["n_variables_bqm"] for d in details),
        }
    return out


# --------------------------------------------------------------------------
# Sections attendues -- l'absence vaut liste de taches
# --------------------------------------------------------------------------
ATTENDU = {
    "sweeps": "Balayage num_sweeps par granularite",
    "surcout": "Perte de decomposition vs perte de recuit",
    "baseline_jointe": "QUBO joint sans decomposition (867 variables)",
    "campagne": "20 runs par n, avec intervalles de confiance",
    "balayage_emax": "Faisabilite en fonction du plafond carbone",
    "plafond_n": "Granularite maximale avant rupture carbone",
}


def main():
    if not FIXTURE.exists():
        raise SystemExit(f"Instance introuvable : {FIXTURE}")

    data = json.loads(FIXTURE.read_text())
    mesures = json.loads(MEASURES.read_text()) if MEASURES.exists() else {}

    print("Instance et partitions...")
    doc = {
        "instance": instance(data),
        "partitions": partitions(data, [2, 3, 4]),
        "mesures": mesures,
        "manquant": {k: v for k, v in ATTENDU.items() if k not in mesures},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"\nEcrit dans {OUT}")

    if doc["manquant"]:
        print("\nSections encore vides :")
        for k, v in doc["manquant"].items():
            print(f"  - {k:18s} {v}")
    else:
        print("\nToutes les sections sont renseignees.")


if __name__ == "__main__":
    main()