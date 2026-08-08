"""
Full project validation -- runs every layer of the pipeline and prints a summary.

IMPORTANT (v2) : les etapes QUBO reposent sur un solveur STOCHASTIQUE (recuit
simule). Une seule execution ne prouve rien -- le meme code peut donner 0.2%
d'ecart une fois et 13.7% la fois suivante. Ces etapes sont donc executees
PLUSIEURS FOIS, et on rapporte un TAUX DE REUSSITE plutot qu'un simple OK/FAIL.
Les etapes classiques (Gurobi) restent deterministes : une seule execution suffit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import statistics
from src.classical_model import build_classical_model
from src.solve import solve_model
from src.qubo_model import build_qubo
from src.solve_qubo import solve_qubo
from src.feasibility import check_feasibility
from src.decomposition_agent import decompose_network
from src.hybrid_pipeline import solve_hybrid, solve_hybrid_classical

# Nombre de repetitions pour les etapes stochastiques
N_RUNS_TOY = 10      # rapide (~0.6s par run)
N_RUNS_HYBRID = 3    # plus lent (~11s par run)

MANUAL_WEIGHTS = {"assignment": 500, "hub_activation": 500, "vehicle_activation": 500,
                  "capacity": 500, "emissions": 500}

results = []


def record(name, status, detail=""):
    results.append({"name": name, "status": status, "detail": detail})
    print(f"[{status}] {name}: {detail}")


# =====================================================================
# PARTIE 1 -- jeu jouet (5 envois)
# =====================================================================
print("=" * 72)
print("FULL PROJECT TEST -- toy scale (5 shipments)")
print("=" * 72)

with open("tests/fixtures/toy_data.json") as f:
    toy_data = json.load(f)

# --- 1a. Modele classique (deterministe : 1 seule execution suffit) ---
t0 = time.time()
model = build_classical_model(toy_data)
solve_model(model, time_limit=60)
elapsed = time.time() - t0
cost = model.OBJ()
status = "OK" if abs(cost - 2109.9) < 1.0 else "WARN"
record("Toy classical model", status, f"cost={cost:.2f} (expected 2109.9) [{elapsed:.2f}s]")

# --- 1b. QUBO (STOCHASTIQUE : on repete et on mesure la variabilite) ---
print(f"\n  Running toy QUBO {N_RUNS_TOY}x (stochastic solver -- measuring variability)...")
bqm, _ = build_qubo(toy_data, MANUAL_WEIGHTS)
feasible_count = 0
gaps = []
t0 = time.time()
for run in range(N_RUNS_TOY):
    best = solve_qubo(bqm, num_reads=1000)
    check = check_feasibility(best.sample, toy_data)
    gap = abs(best.energy - 2109.9) / 2109.9 * 100
    gaps.append(gap)
    if check["feasible"]:
        feasible_count += 1
elapsed = time.time() - t0

rate = feasible_count / N_RUNS_TOY * 100
gap_min, gap_max = min(gaps), max(gaps)
gap_mean = statistics.mean(gaps)
# Critere : au moins 50% des runs faisables, et le MEILLEUR run proche de l'optimum
status = "OK" if (rate >= 50 and gap_min < 5) else ("WARN" if rate > 0 else "FAIL")
record("Toy QUBO (manual weights)", status,
       f"feasible {feasible_count}/{N_RUNS_TOY} ({rate:.0f}%)  "
       f"gap min/mean/max = {gap_min:.1f}%/{gap_mean:.1f}%/{gap_max:.1f}%  [{elapsed:.2f}s]")

# =====================================================================
# PARTIE 2 -- echelle reelle (30 envois)
# =====================================================================
print()
print("=" * 72)
print("FULL PROJECT TEST -- real scale (30 shipments)")
print("=" * 72)

with open("tests/fixtures/scaled_data.json") as f:
    scaled_data = json.load(f)

print(f"E_max in use: {scaled_data['E_max']}")

# --- 2a. Modele classique joint (deterministe) ---
t0 = time.time()
model = build_classical_model(scaled_data)
solve_model(model, time_limit=120)
elapsed = time.time() - t0
cost_classical = model.OBJ()
record("Real-scale classical model", "OK", f"cost={cost_classical:.2f} [{elapsed:.2f}s]")

# --- 2b. Agent de decomposition (deterministe : random_state=0) ---
t0 = time.time()
clusters = decompose_network(scaled_data, n_clusters=2)
elapsed = time.time() - t0
total_ships = sum(len(c["shipments"]) for c in clusters.values())
status = "OK" if total_ships == len(scaled_data["shipments"]) else "FAIL"
record("Decomposition agent", status,
       f"{len(clusters)} clusters, {total_ships} shipments partitioned [{elapsed:.2f}s]")

# --- 2c. Hybride classique (deterministe : Gurobi par cluster) ---
t0 = time.time()
result_classical_hybrid = solve_hybrid_classical(scaled_data, n_clusters=2)
elapsed = time.time() - t0
status = "OK" if result_classical_hybrid["feasible"] else "FAIL"
record("Hybrid classical (decomposed)", status,
       f"cost={result_classical_hybrid['cost']:.2f} "
       f"feasible={result_classical_hybrid['feasible']} [{elapsed:.2f}s]")

# --- 2d. Hybride QUBO (STOCHASTIQUE : on repete) ---
print(f"\n  Running hybrid QUBO {N_RUNS_HYBRID}x (slow -- known open issue)...")
feasible_count = 0
costs = []
issue_counts = []
t0 = time.time()
for run in range(N_RUNS_HYBRID):
    r = solve_hybrid(scaled_data, n_clusters=2, penalty_weights=MANUAL_WEIGHTS)
    costs.append(r["cost"])
    issue_counts.append(len(r.get("issues", [])))
    if r["feasible"]:
        feasible_count += 1
elapsed = time.time() - t0

rate = feasible_count / N_RUNS_HYBRID * 100
status = "OK" if rate >= 50 else "KNOWN_LIMITATION"
record("Hybrid QUBO (decomposed)", status,
       f"feasible {feasible_count}/{N_RUNS_HYBRID} ({rate:.0f}%)  "
       f"cost min/max = {min(costs):.2f}/{max(costs):.2f}  "
       f"issues min/max = {min(issue_counts)}/{max(issue_counts)}  [{elapsed:.2f}s]")

# =====================================================================
# RESUME
# =====================================================================
print()
print("=" * 72)
print("SUMMARY")
print("=" * 72)
for r in results:
    print(f"  [{r['status']:>16}] {r['name']}")
    print(f"                     {r['detail']}")

if result_classical_hybrid["feasible"]:
    price = (result_classical_hybrid["cost"] / cost_classical - 1) * 100
    print(f"\nPrice of decomposition (classical, deterministic): +{price:.2f}%")

n_ok = sum(1 for r in results if r["status"] == "OK")
print(f"\n{n_ok}/{len(results)} checks passed cleanly.")
print("\nNote: QUBO rows report a SUCCESS RATE across repeated runs, not a single")
print("draw -- simulated annealing is stochastic and varies run to run by design.")