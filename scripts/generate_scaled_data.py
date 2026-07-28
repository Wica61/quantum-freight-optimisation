# scripts/generate_scaled_data.py
"""
Genere une instance realiste : 30 envois, 3 hubs, 12 vehicules (flotte reelle,
pas 1 seul vehicule par type). Les fenetres de temps sont construites pour
etre garanties faisables -- pas laissees au hasard puis verifiees apres coup.
"""
import json
import math
import random
from pathlib import Path

random.seed(42)  # reproductible -- ne changez pas sans le documenter

N_SHIPMENTS = 30
N_SLOTS = 6

HUBS = {
    "h1": {"x": 0, "y": 0, "cost": 1000},
    "h2": {"x": 15, "y": 5, "cost": 1100},
    "h3": {"x": 8, "y": 18, "cost": 1200},
}

VEHICLES = {}
for k in range(1, 7):
    VEHICLES[f"van_{k}"] = {"capacity": 15, "fixed_cost": 300, "rate": 8, "handling": 3, "emission_rate": 1.5, "speed": 40}
for k in range(1, 7):
    VEHICLES[f"truck_{k}"] = {"capacity": 20, "fixed_cost": 400, "rate": 6, "handling": 2, "emission_rate": 2.5, "speed": 60}

SHIPMENTS = {
    f"s{i}": {"weight": random.randint(3, 10), "x": random.uniform(0, 20), "y": random.uniform(0, 20)}
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
                travel_time = d / veh["speed"]
                arrival_slot[key] = min(N_SLOTS - 1, math.floor(travel_time * SLOT_SCALE))

    # Fenetre garantie faisable : on tire une combinaison de reference pour
    # chaque envoi, on lit son creneau reel, et on construit la fenetre autour
    # -- au moins CETTE combinaison restera valide, quoi qu'il arrive.
    windows = {}
    for i in SHIPMENTS:
        h_ref, v_ref = random.choice([(h, v) for h in HUBS for v in VEHICLES])
        ref_slot = arrival_slot[f"{i}|{h_ref}|{v_ref}"]
        slack = random.choice([0, 1])
        windows[i] = (max(0, ref_slot - slack), min(N_SLOTS - 1, ref_slot + slack))
        SHIPMENTS[i]["e"], SHIPMENTS[i]["l"] = windows[i]

    valid_combinations = {
        i: [f"{h}|{v}" for h in HUBS for v in VEHICLES
            if SHIPMENTS[i]["e"] <= arrival_slot[f"{i}|{h}|{v}"] <= SHIPMENTS[i]["l"]]
        for i in SHIPMENTS
    }

    return {
        "shipments": SHIPMENTS, "hubs": HUBS, "vehicles": VEHICLES,
        "transport_cost": transport_cost, "emission": emission,
        "arrival_slot": arrival_slot, "valid_combinations": valid_combinations,
        "E_max": None,  # calibre a l'etape 14.4 -- ne pas deviner
    }

if __name__ == "__main__":
    data = build_dataset()

    empty = [i for i, c in data["valid_combinations"].items() if not c]
    assert not empty, f"Envois sans combinaison valide (ne devrait jamais arriver) : {empty}"

    total_weight = sum(s["weight"] for s in data["shipments"].values())
    total_capacity = sum(v["capacity"] for v in data["vehicles"].values())
    print(f"Poids total : {total_weight}  |  Capacité flotte totale : {total_capacity}  "
          f"|  Marge : {total_capacity - total_weight}")
    assert total_capacity > total_weight, "Flotte sous-dimensionnée dès la génération"

    n_x = sum(len(c) for c in data["valid_combinations"].values())
    print(f"Variables x prévues dans le QUBO : {n_x}")

    out_path = Path("tests/fixtures/scaled_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Instance réaliste écrite dans {out_path}")