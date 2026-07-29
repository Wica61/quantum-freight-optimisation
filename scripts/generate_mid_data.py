"""
Genere une instance INTERMEDIAIRE, concue pour que TOUT le pipeline fonctionne :
  - assez grande pour que la decomposition ait un sens (contrairement au jouet
    a 5 envois, ou 2 clusters laissent 1 seul vehicule chacun -> infaisable)
  - assez petite pour que le recuit simule ait une vraie chance (70 variables x,
    contre 468 pour l'instance a 30 envois)

Parametres retenus apres test : 12 envois, 2 hubs, 2 vans + 2 trucks.
  -> marge de capacite 40% (la contrainte contraint vraiment, sans etre etouffante)
  -> les 2 clusters restent individuellement faisables apres decomposition

Le script calibre AUSSI E_max lui-meme, dans le bon ordre (decomposer d'abord,
mesurer les planchers reels ensuite, calibrer enfin) -- ce qui evite le piege
rencontre sur l'instance a 30 envois, ou E_max avait ete cale sur un plancher
joint theorique que la decomposition ne pouvait pas atteindre.
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

SEED = 7
N_SHIPMENTS = 12
N_SLOTS = 6
N_CLUSTERS = 2
MARGIN = 1.20          # 20% au-dessus des planchers decomposes reels
OUT = Path("tests/fixtures/mid_data.json")

random.seed(SEED)

HUBS = {
    "h1": {"x": 0, "y": 0, "cost": 800},
    "h2": {"x": 12, "y": 10, "cost": 900},
}

VEHICLES = {}
for k in range(1, 3):
    VEHICLES[f"van_{k}"] = {"capacity": 15, "fixed_cost": 300, "rate": 8,
                            "handling": 3, "emission_rate": 1.5, "speed": 40}
for k in range(1, 3):
    VEHICLES[f"truck_{k}"] = {"capacity": 20, "fixed_cost": 400, "rate": 6,
                              "handling": 2, "emission_rate": 2.5, "speed": 60}

SHIPMENTS = {
    f"s{i}": {"weight": random.randint(3, 9),
              "x": random.uniform(0, 14),
              "y": random.uniform(0, 12)}
    for i in range(1, N_SHIPMENTS + 1)
}


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


SLOT_SCALE = 15


def build_dataset():
    transport_cost, emission, arrival_slot = {}, {}, {}
    for i, sh in SHIPMENTS.items():
        for h, hb in HUBS.items():
            d = dist(sh, hb)
            for v, veh in VEHICLES.items():
                key = f"{i}|{h}|{v}"
                transport_cost[key] = round(veh["rate"] * d + veh["handling"] * sh["weight"], 2)
                emission[key] = round(veh["emission_rate"] * d, 2)
                arrival_slot[key] = min(N_SLOTS - 1, math.floor((d / veh["speed"]) * SLOT_SCALE))

    # Fenetres garanties faisables par construction : on tire une combinaison de
    # reference par envoi et on batit la fenetre autour de SON creneau reel.
    for i in SHIPMENTS:
        h_ref, v_ref = random.choice([(h, v) for h in HUBS for v in VEHICLES])
        ref = arrival_slot[f"{i}|{h_ref}|{v_ref}"]
        slack = random.choice([1, 1, 2])
        SHIPMENTS[i]["e"] = max(0, ref - slack)
        SHIPMENTS[i]["l"] = min(N_SLOTS - 1, ref + slack)

    valid_combinations = {
        i: [f"{h}|{v}" for h in HUBS for v in VEHICLES
            if SHIPMENTS[i]["e"] <= arrival_slot[f"{i}|{h}|{v}"] <= SHIPMENTS[i]["l"]]
        for i in SHIPMENTS
    }

    return {
        "shipments": SHIPMENTS, "hubs": HUBS, "vehicles": VEHICLES,
        "transport_cost": transport_cost, "emission": emission,
        "arrival_slot": arrival_slot, "valid_combinations": valid_combinations,
        "E_max": 10 ** 9,   # placeholder non contraignant -- calibre plus bas
    }


def emissions_floor(sub_data):
    """Plancher d'emissions REEL de ce sous-probleme, avec ses propres vehicules."""
    d = dict(sub_data)
    d["E_max"] = 10 ** 9
    model = build_classical_model(d)
    model.del_component(model.OBJ)

    def rule(mdl):
        return sum(d["emission"][f"{i}|{h}|{v}"] * mdl.x[i, h, v] for (i, h, v) in mdl.VALID)

    model.OBJ = Objective(rule=rule, sense=minimize)
    solve_model(model, time_limit=60)
    return model.OBJ()


if __name__ == "__main__":
    data = build_dataset()

    # --- controles de coherence avant tout calcul lourd ---
    empty = [i for i, c in data["valid_combinations"].items() if not c]
    assert not empty, f"Envois sans combinaison valide : {empty}"

    total_weight = sum(s["weight"] for s in data["shipments"].values())
    total_capacity = sum(v["capacity"] for v in data["vehicles"].values())
    n_x = sum(len(c) for c in data["valid_combinations"].values())
    print(f"Envois={len(data['shipments'])}  hubs={len(data['hubs'])}  "
          f"vehicules={len(data['vehicles'])}")
    print(f"Poids total={total_weight}  capacite totale={total_capacity}  "
          f"marge={(total_capacity/total_weight-1)*100:.0f}%")
    print(f"Variables x prevues (QUBO joint) = {n_x}")
    assert total_capacity > total_weight, "Flotte sous-dimensionnee"

    # --- calibration d'E_max DANS LE BON ORDRE ---
    print(f"\nCalibration d'E_max pour {N_CLUSTERS} clusters...")
    clusters = decompose_network(data, n_clusters=N_CLUSTERS)

    floors = {}
    for cid, sub in clusters.items():
        w = sum(sub["shipments"][i]["weight"] for i in sub["shipments"])
        cap = sum(sub["vehicles"][v]["capacity"] for v in sub["vehicles"])
        assert cap >= w, (f"Cluster {cid} sous-dote : capacite {cap} < poids {w} "
                          f"-- decomposition non viable, reduire N_CLUSTERS")
        floors[cid] = emissions_floor(sub)
        print(f"  cluster {cid}: {len(sub['shipments'])} envois, {len(sub['vehicles'])} "
              f"vehicules, capacite={cap} (poids={w}), plancher emissions={floors[cid]:.2f}")

    floor_sum = sum(floors.values())
    data["E_max"] = round(floor_sum * MARGIN, 2)
    print(f"\nSomme des planchers decomposes = {floor_sum:.2f}")
    print(f"E_max calibre (+{int((MARGIN-1)*100)}%) = {data['E_max']}")

    # --- verification finale : le modele joint reste-t-il resoluble ? ---
    print("\nVerification du modele joint avec E_max calibre...")
    model = build_classical_model(data)
    solve_model(model, time_limit=120)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ Instance intermediaire ecrite dans {OUT}")