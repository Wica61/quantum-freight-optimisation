# scripts/recalibrate_for_decomposition.py
"""
Corrige le calibrage de l'etape 14.4 : plutot que de partir du plancher JOINT
(qui suppose une flexibilite totale qu'aucune decomposition ne conserve), on
decompose D'ABORD (avec un plafond non contraignant), on mesure le plancher
REEL de chaque cluster (avec ses vehicules deja alloues), et on calibre
E_max_global au-dessus de la somme de ces planchers reels.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from pyomo.environ import Objective, minimize
from src.decomposition_agent import decompose_network
from src.classical_model import build_classical_model
from src.solve import solve_model

N_CLUSTERS = 2  # fige ici -- 3 clusters s'est revele trop agressif pour cette instance (voir etape 14.5)
MARGIN = 1.20   # 20% au-dessus de ce que la decomposition peut reellement atteindre

with open("tests/fixtures/scaled_data.json") as f:
    data = json.load(f)

data_temp = dict(data)
data_temp["E_max"] = 10**9  # non contraignant -- sert juste a obtenir le decoupage
clusters = decompose_network(data_temp, n_clusters=N_CLUSTERS)

floors = {}
for cluster_id, sub_data in clusters.items():
    model = build_classical_model(sub_data)
    model.del_component(model.OBJ)
    def min_emissions_rule(mdl):
        return sum(sub_data["emission"][f"{i}|{h}|{v}"] * mdl.x[i, h, v] for (i, h, v) in mdl.VALID)
    model.OBJ = Objective(rule=min_emissions_rule, sense=minimize)
    solve_model(model, time_limit=60)
    floors[cluster_id] = model.OBJ()
    print(f"Plancher réel cluster {cluster_id} ({len(sub_data['shipments'])} envois) : {floors[cluster_id]:.2f}")

floor_sum = sum(floors.values())
E_max_new = round(floor_sum * MARGIN, 2)
print(f"\nSomme des planchers décomposés : {floor_sum:.2f}")
print(f"Nouveau E_max_global (planchers décomposés + {int((MARGIN-1)*100)}%) : {E_max_new}")

data["E_max"] = E_max_new
with open("tests/fixtures/scaled_data.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"✓ E_max mis à jour dans scaled_data.json (remplace l'ancienne valeur 614.5)")