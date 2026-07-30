"""
Regenere l'instance a 30 envois avec une flotte SUFFISANTE.

Justification chiffree (diagnostic du 30/07) : sur 9 clusters testes, le QUBO
reussit systematiquement au-dela de 20% de marge de capacite et echoue
systematiquement en dessous de 13%. L'instance actuelle n'a que 14.8% de marge
GLOBALE (183 de poids / 210 de capacite) -- il est donc mathematiquement
impossible de donner 20% de marge a chaque cluster. L'instance est hors du
domaine d'applicabilite de la methode, independamment du reglage.

Correctif : 8 vans + 8 camions -> capacite 280, marge 53%.
Les 30 ENVOIS RESTENT IDENTIQUES (meme graine, meme ordre de tirage), seule la
flotte change -- les resultats restent donc comparables a l'ancienne instance.

Les fenetres de temps sont DETERMINISTES (hub le plus proche a vitesse van,
tolerance +/-1 creneau) et non tirees au hasard parmi les combinaisons
(hub, vehicule) : sinon changer la flotte modifierait aussi les fenetres.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import math
import random

from pyomo.environ import Objective, minimize
from src.classical_model import build_classical_model
from src.solve import solve_model
from src.decomposition_agent import decompose_network

N_VANS, N_TRUCKS = 8, 8
N_CLUSTERS_CIBLE = 4        # granularite visee pour la calibration d'E_max
MARGE_EMISSIONS = 1.25
N_SLOTS = 6
SLOT_SCALE = 15
VAN_SPEED = 40
OUT = Path("tests/fixtures/scaled_data_v2.json")

HUBS = {
    "h1": {"x": 0,  "y": 0,  "cost": 1000},
    "h2": {"x": 15, "y": 5,  "cost": 1100},
    "h3": {"x": 8,  "y": 18, "cost": 1200},
}

VEHICLES = {}
for k in range(1, N_VANS + 1):
    VEHICLES[f"van_{k}"] = {"capacity": 15, "fixed_cost": 300, "rate": 8,
                            "handling": 3, "emission_rate": 1.5, "speed": 40}
for k in range(1, N_TRUCKS + 1):
    VEHICLES[f"truck_{k}"] = {"capacity": 20, "fixed_cost": 400, "rate": 6,
                              "handling": 2, "emission_rate": 2.5, "speed": 60}

# RNG dedie : reproduit EXACTEMENT les 30 envois d'origine (seed 42, meme ordre)
rng = random.Random(42)
SHIPMENTS = {
    f"s{i}": {"weight": rng.randint(3, 10),
              "x": rng.uniform(0, 20),
              "y": rng.uniform(0, 20)}
    for i in range(1, 31)
}


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def build():
    for i, sh in SHIPMENTS.items():
        d_min = min(dist(sh, hb) for hb in HUBS.values())
        slot = min(N_SLOTS - 1, math.floor((d_min / VAN_SPEED) * SLOT_SCALE))
        sh["e"] = max(0, slot - 1)
        sh["l"] = min(N_SLOTS - 1, slot + 1)

    transport_cost, emission, arrival_slot = {}, {}, {}
    for i, sh in SHIPMENTS.items():
        for h, hb in HUBS.items():
            d = dist(sh, hb)
            for v, veh in VEHICLES.items():
                key = f"{i}|{h}|{v}"
                transport_cost[key] = round(veh["rate"] * d + veh["handling"] * sh["weight"], 2)
                emission[key] = round(veh["emission_rate"] * d, 2)
                arrival_slot[key] = min(N_SLOTS - 1,
                                        math.floor((d / veh["speed"]) * SLOT_SCALE))

    valid = {
        i: [f"{h}|{v}" for h in HUBS for v in VEHICLES
            if SHIPMENTS[i]["e"] <= arrival_slot[f"{i}|{h}|{v}"] <= SHIPMENTS[i]["l"]]
        for i in SHIPMENTS
    }

    return {"shipments": SHIPMENTS, "hubs": HUBS, "vehicles": VEHICLES,
            "transport_cost": transport_cost, "emission": emission,
            "arrival_slot": arrival_slot, "valid_combinations": valid,
            "E_max": 10 ** 9}


def plancher(sub):
    d = dict(sub)
    d["E_max"] = 10 ** 9
    m = build_classical_model(d)
    m.del_component(m.OBJ)

    def regle(mdl):
        return sum(d["emission"][f"{i}|{h}|{v}"] * mdl.x[i, h, v]
                   for (i, h, v) in mdl.VALID)

    m.OBJ = Objective(rule=regle, sense=minimize)
    solve_model(m, time_limit=60)
    try:
        return m.OBJ()
    except Exception:
        return None


if __name__ == "__main__":
    data = build()

    orph = [i for i, c in data["valid_combinations"].items() if not c]
    assert not orph, f"Envois sans combinaison : {orph}"

    poids = sum(s["weight"] for s in data["shipments"].values())
    cap = sum(v["capacity"] for v in data["vehicles"].values())
    nx = sum(len(c) for c in data["valid_combinations"].values())
    print(f"Envois : {len(data['shipments'])} (poids total {poids})")
    print(f"Flotte : {N_VANS} vans + {N_TRUCKS} trucks = {len(VEHICLES)} vehicules, "
          f"capacite {cap}")
    print(f"Marge de capacite globale : {(cap/poids-1)*100:.1f}%  "
          f"(seuil de reussite mesure : >20%)")
    print(f"Variables x : {nx}")

    # --- calibration d'E_max pour la granularite visee ---
    print(f"\nCalibration d'E_max pour n_clusters={N_CLUSTERS_CIBLE}...")
    clusters = decompose_network(data, n_clusters=N_CLUSTERS_CIBLE)
    planchers = {}
    for cid, sub in clusters.items():
        p = sum(sub["shipments"][i]["weight"] for i in sub["shipments"])
        c = sum(sub["vehicles"][v]["capacity"] for v in sub["vehicles"])
        f = plancher(sub)
        planchers[cid] = f
        marge_c = (c / p - 1) * 100 if p else 0
        etat = "OK" if marge_c > 20 else "MARGE FAIBLE"
        print(f"  cluster {cid}: {len(sub['shipments']):>2} envois, "
              f"{len(sub['vehicles']):>2} veh, poids {p:>3}/{c:<3} "
              f"(marge {marge_c:>5.1f}%) [{etat}], plancher "
              f"{f if f is None else round(f, 2)}")

    assert all(f is not None for f in planchers.values()), \
        "Un cluster est mal forme -- reduire N_CLUSTERS_CIBLE"

    somme = sum(planchers.values())
    data["E_max"] = round(somme * MARGE_EMISSIONS, 2)
    print(f"\nSomme des planchers = {somme:.2f}")
    print(f"E_max calibre (+{int((MARGE_EMISSIONS-1)*100)}%) = {data['E_max']}")

    print("\nVerification du modele joint...")
    m = build_classical_model(data)
    solve_model(m, time_limit=300)
    try:
        print(f"Optimum joint = {m.OBJ():.2f}")
    except Exception:
        print("ATTENTION : modele joint INFAISABLE -- relever MARGE_EMISSIONS")
        sys.exit(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ Ecrit dans {OUT}")