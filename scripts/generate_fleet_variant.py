"""
Genere une variante de l'instance a 12 envois avec une FLOTTE PARAMETRABLE,
en gardant EXACTEMENT les memes marchandises a transporter.

Usage :
    python scripts/generate_fleet_variant.py 2 2     # 2 vans + 2 trucks (reference)
    python scripts/generate_fleet_variant.py 4 4
    python scripts/generate_fleet_variant.py 6 6

Point technique important : les fenetres de temps sont DETERMINISTES (calculees
a partir du hub le plus proche a vitesse van, avec une tolerance de +/-1
creneau) et non tirees au hasard parmi les combinaisons (hub, vehicule).
Sinon, changer le nombre de vehicules modifierait la sequence aleatoire et donc
les fenetres -- les envois ne seraient plus comparables d'une flotte a l'autre.

Les 12 envois (poids, positions, fenetres) sont donc IDENTIQUES quelle que
soit la flotte. Poids total : 50.
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

N_SHIPMENTS = 12
N_SLOTS = 6
SLOT_SCALE = 15
VAN_SPEED = 40          # vitesse de reference pour les fenetres (la plus lente)
N_CLUSTERS = 2
MARGIN = 1.20
SEED = 7

n_vans = int(sys.argv[1]) if len(sys.argv) > 2 else 2
n_trucks = int(sys.argv[2]) if len(sys.argv) > 2 else 2
OUT = Path(f"tests/fixtures/mid_data_{n_vans}v{n_trucks}t.json")

HUBS = {
    "h1": {"x": 0, "y": 0, "cost": 800},
    "h2": {"x": 12, "y": 10, "cost": 900},
}

VEHICLES = {}
for k in range(1, n_vans + 1):
    VEHICLES[f"van_{k}"] = {"capacity": 15, "fixed_cost": 300, "rate": 8,
                            "handling": 3, "emission_rate": 1.5, "speed": 40}
for k in range(1, n_trucks + 1):
    VEHICLES[f"truck_{k}"] = {"capacity": 20, "fixed_cost": 400, "rate": 6,
                              "handling": 2, "emission_rate": 2.5, "speed": 60}

# RNG dedie aux envois : la sequence ne depend PAS de la flotte
rng = random.Random(SEED)
SHIPMENTS = {
    f"s{i}": {"weight": rng.randint(3, 9),
              "x": rng.uniform(0, 14),
              "y": rng.uniform(0, 12)}
    for i in range(1, N_SHIPMENTS + 1)
}


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def build_dataset():
    # Fenetres DETERMINISTES : autour du creneau d'arrivee depuis le hub le plus
    # proche a vitesse van, avec une tolerance de +/- 1 creneau.
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
                arrival_slot[key] = min(N_SLOTS - 1, math.floor((d / veh["speed"]) * SLOT_SCALE))

    valid_combinations = {
        i: [f"{h}|{v}" for h in HUBS for v in VEHICLES
            if SHIPMENTS[i]["e"] <= arrival_slot[f"{i}|{h}|{v}"] <= SHIPMENTS[i]["l"]]
        for i in SHIPMENTS
    }

    return {
        "shipments": SHIPMENTS, "hubs": HUBS, "vehicles": VEHICLES,
        "transport_cost": transport_cost, "emission": emission,
        "arrival_slot": arrival_slot, "valid_combinations": valid_combinations,
        "E_max": 10 ** 9,
    }


def emissions_floor(sub_data):
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

    empty = [i for i, c in data["valid_combinations"].items() if not c]
    assert not empty, f"Envois sans combinaison valide : {empty}"

    total_weight = sum(s["weight"] for s in data["shipments"].values())
    total_capacity = sum(v["capacity"] for v in data["vehicles"].values())
    n_x = sum(len(c) for c in data["valid_combinations"].values())
    print(f"Flotte : {n_vans} vans + {n_trucks} trucks = {len(VEHICLES)} vehicules")
    print(f"Envois : {N_SHIPMENTS}, poids total = {total_weight} (identique a "
          f"toutes les variantes)")
    print(f"Capacite totale = {total_capacity}, marge = {total_capacity - total_weight} "
          f"({(total_capacity/total_weight-1)*100:.0f}%)")
    print(f"Variables x prevues = {n_x}")
    assert total_capacity > total_weight, "Flotte sous-dimensionnee"

    print(f"\nCalibration d'E_max sur {N_CLUSTERS} clusters...")
    clusters = decompose_network(data, n_clusters=N_CLUSTERS)
    floors = {}
    for cid, sub in clusters.items():
        w = sum(sub["shipments"][i]["weight"] for i in sub["shipments"])
        cap = sum(sub["vehicles"][v]["capacity"] for v in sub["vehicles"])
        assert cap >= w, (f"Cluster {cid} sous-dote : capacite {cap} < poids {w}")
        floors[cid] = emissions_floor(sub)
        print(f"  cluster {cid}: {len(sub['shipments'])} envois, "
              f"{len(sub['vehicles'])} vehicules, capacite={cap} (poids={w}), "
              f"plancher={floors[cid]:.2f}")

    data["E_max"] = round(sum(floors.values()) * MARGIN, 2)
    print(f"\nE_max calibre = {data['E_max']}")

    print("\nVerification du modele joint...")
    model = build_classical_model(data)
    solve_model(model, time_limit=180)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ Ecrit dans {OUT}")