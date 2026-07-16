# scripts/generate_toy_data.py
"""
Genere un jeu de donnees jouet complet : 5 envois, 2 hubs, 2 vehicules.
Toutes les combinaisons cout/emission/creneau sont calculees, pas laissees
a completer a la main.
"""
import json
import math
from pathlib import Path

SHIPMENTS = {
    "s1": {"weight": 5, "x": 2, "y": 3, "e": 1, "l": 1},
    "s2": {"weight": 8, "x": 5, "y": 1, "e": 0, "l": 1},
    "s3": {"weight": 3, "x": 1, "y": 4, "e": 1, "l": 1},
    "s4": {"weight": 6, "x": 6, "y": 5, "e": 0, "l": 1},
    "s5": {"weight": 4, "x": 3, "y": 2, "e": 1, "l": 1},
}
HUBS = {
    "h1": {"x": 0, "y": 0, "cost": 1000},
    "h2": {"x": 6, "y": 6, "cost": 1200},
}
VEHICLES = {
    "v1": {"capacity": 15, "fixed_cost": 300, "rate": 8, "handling": 3, "emission_rate": 1.5, "speed": 40},
    "v2": {"capacity": 20, "fixed_cost": 400, "rate": 6, "handling": 2, "emission_rate": 2.5, "speed": 60},
}
SLOT_SCALE = 15  # calibre empiriquement pour etaler les creneaux sur 0..3 -- voir 3.2

def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

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
                arrival_slot[key] = min(3, math.floor(travel_time * SLOT_SCALE))

    # Ne garder que les combinaisons (h,v) qui respectent la fenetre de l'envoi --
    # c'est cette liste qui definit les variables x_i_h_v effectivement creees
    # dans le modele (etape 4), pas de variable = pas de violation possible.
    valid_combinations = {}
    for i, sh in SHIPMENTS.items():
        valid_combinations[i] = [
            f"{h}|{v}"
            for h in HUBS for v in VEHICLES
            if sh["e"] <= arrival_slot[f"{i}|{h}|{v}"] <= sh["l"]
        ]

    return {
        "shipments": SHIPMENTS, "hubs": HUBS, "vehicles": VEHICLES,
        "transport_cost": transport_cost, "emission": emission,
        "arrival_slot": arrival_slot, "valid_combinations": valid_combinations,
        "E_max": 45.4,  # calibre a l'etape 3.2 -- voir la justification
    }

if __name__ == "__main__":
    data = build_dataset()
    out_path = Path("tests/fixtures/toy_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Jeu de données jouet écrit dans {out_path}")
    for i, combos in data["valid_combinations"].items():
        print(f"  {i}: {len(combos)} combinaison(s) valide(s) sur 4 -> {combos}")


"""
python scripts/generate_toy_data.py

"""