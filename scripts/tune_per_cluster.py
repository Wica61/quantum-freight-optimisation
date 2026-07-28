# scripts/tune_per_cluster.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src.decomposition_agent import decompose_network
from src.penalty_tuning import tune_penalties

with open("tests/fixtures/scaled_data.json") as f:
    data = json.load(f)

clusters = decompose_network(data, n_clusters=2)

learned_weights_per_cluster = {}
for cluster_id, sub_data in clusters.items():
    print(f"--- Apprentissage des poids pour le cluster {cluster_id} "
          f"({len(sub_data['shipments'])} envois, {len(sub_data['vehicles'])} véhicules) ---")
    learned_weights_per_cluster[cluster_id] = tune_penalties(sub_data, n_trials=10)

with open("results/learned_weights_per_cluster.json", "w") as f:
    json.dump(learned_weights_per_cluster, f, indent=2)
print("✓ Poids appris par cluster sauvegardés")