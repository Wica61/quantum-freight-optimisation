# scripts/calibrate_emissions_cap.py
"""
Calibre E_max sans enumeration brute-force : resout le modele classique une fois
en minimisant le cout (objectif normal, sans plafond) pour voir le niveau
d'emissions "naturel", puis une seconde fois en minimisant les EMISSIONS
elles-memes (meme structure de contraintes, objectif different) pour trouver
le plancher reellement atteignable. E_max est cale entre les deux.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 🔧 voir note ci-dessous

import json
from pyomo.environ import Objective, minimize
from src.classical_model import build_classical_model
from src.solve import solve_model

with open("tests/fixtures/scaled_data.json") as f:
    data = json.load(f)

# --- Run 1 : cout minimal, emissions non contraintes ---
data_no_cap = dict(data)
data_no_cap["E_max"] = 10**9  # plafond volontairement non contraignant
model_cost = build_classical_model(data_no_cap)
solve_model(model_cost, time_limit=120)
emissions_at_cost_optimal = sum(
    data["emission"][f"{i}|{h}|{v}"] * model_cost.x[i, h, v].value
    for (i, h, v) in model_cost.VALID
)
print(f"Coût minimal (sans plafond) : {model_cost.OBJ():.2f}")
print(f"Émissions à ce coût optimal : {emissions_at_cost_optimal:.2f}")

# --- Run 2 : emissions minimales, cout ignore ---
model_emissions = build_classical_model(data_no_cap)
model_emissions.del_component(model_emissions.OBJ)  # remplace l'objectif
def min_emissions_rule(mdl):
    return sum(data["emission"][f"{i}|{h}|{v}"] * mdl.x[i, h, v] for (i, h, v) in mdl.VALID)
model_emissions.OBJ = Objective(rule=min_emissions_rule, sense=minimize)
solve_model(model_emissions, time_limit=120)
floor_emissions = model_emissions.OBJ()
print(f"Plancher d'émissions réellement atteignable : {floor_emissions:.2f}")

# --- Calibrage ---
E_max = round(floor_emissions + 0.5 * (emissions_at_cost_optimal - floor_emissions), 2)
print(f"\nE_max calibré : {E_max}")
print(f"(à mi-chemin entre le plancher {floor_emissions:.2f} et le niveau naturel {emissions_at_cost_optimal:.2f})")

data["E_max"] = E_max
with open("tests/fixtures/scaled_data.json", "w") as f:
    json.dump(data, f, indent=2)
print("✓ E_max mis à jour dans scaled_data.json")