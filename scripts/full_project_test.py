"""
Full project validation -- runs every layer of the pipeline in one pass and
prints a summary. Intended to be rerun after any change, to see the whole
project's health at a glance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from src.classical_model import build_classical_model
from src.solve import solve_model
from src.qubo_model import build_qubo
from src.solve_qubo import solve_qubo
from src.feasibility import check_feasibility
from src.decomposition_agent import decompose_network
from src.hybrid_pipeline import solve_hybrid, solve_hybrid_classical

results = []


def record(name, status, detail=""):
    results.append({"name": name, "status": status, "detail": detail})
    icon = "OK" if status == "OK" else ("WARN" if status == "WARN" else "FAIL")
    print(f"[{icon}] {name}: {detail}")


print("=" * 70)
print("FULL PROJECT TEST -- toy scale (5 shipments)")
print("=" * 70)

with open("tests/fixtures/toy_data.json") as f:
    toy_data = json.load(f)

t0 = time.time()
model = build_classical_model(toy_data)
solve_model(model, time_limit=60)
elapsed = time.time() - t0
cost = model.OBJ()
status = "OK" if abs(cost - 2109.9) < 1.0 else "WARN"
record("Toy classical model", status, f"cost={cost:.2f} (expected 2109.9) [{elapsed:.2f}s]")

t0 = time.time()
manual_weights = {"assignment": 500, "hub_activation": 500, "vehicle_activation": 500,
                   "capacity": 500, "emissions": 500}
bqm, _ = build_qubo(toy_data, manual_weights)
best = solve_qubo(bqm, num_reads=1000)
elapsed = time.time() - t0
check = check_feasibility(best.sample, toy_data)
gap = abs(best.energy - 2109.9) / 2109.9 * 100
status = "OK" if check["feasible"] and gap < 10 else "WARN"
record("Toy QUBO (manual weights)", status,
       f"energy={best.energy:.2f} gap={gap:.1f}% feasible={check['feasible']} [{elapsed:.2f}s]")

print()
print("=" * 70)
print("FULL PROJECT TEST -- real scale (30 shipments)")
print("=" * 70)

with open("tests/fixtures/scaled_data.json") as f:
    scaled_data = json.load(f)

print(f"E_max in use: {scaled_data['E_max']}")

t0 = time.time()
model = build_classical_model(scaled_data)
solve_model(model, time_limit=120)
elapsed = time.time() - t0
cost_classical = model.OBJ()
record("Real-scale classical model", "OK", f"cost={cost_classical:.2f} [{elapsed:.2f}s]")

t0 = time.time()
clusters = decompose_network(scaled_data, n_clusters=2)
elapsed = time.time() - t0
total_ships = sum(len(c["shipments"]) for c in clusters.values())
status = "OK" if total_ships == len(scaled_data["shipments"]) else "FAIL"
record("Decomposition agent", status,
       f"{len(clusters)} clusters, {total_ships} shipments partitioned [{elapsed:.2f}s]")

t0 = time.time()
result_classical_hybrid = solve_hybrid_classical(scaled_data, n_clusters=2)
elapsed = time.time() - t0
status = "OK" if result_classical_hybrid["feasible"] else "FAIL"
record("Hybrid classical (decomposed)", status,
       f"cost={result_classical_hybrid['cost']:.2f} feasible={result_classical_hybrid['feasible']} [{elapsed:.2f}s]")

print("\n(next check may take a while -- known open issue on the larger cluster)\n")
t0 = time.time()
result_qubo_hybrid = solve_hybrid(scaled_data, n_clusters=2, penalty_weights=manual_weights)
elapsed = time.time() - t0
status = "OK" if result_qubo_hybrid["feasible"] else "KNOWN_LIMITATION"
record("Hybrid QUBO (decomposed)", status,
       f"cost={result_qubo_hybrid['cost']:.2f} feasible={result_qubo_hybrid['feasible']} "
       f"issues={len(result_qubo_hybrid.get('issues', []))} [{elapsed:.2f}s]")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
for r in results:
    print(f"  [{r['status']:>16}] {r['name']}")

if result_classical_hybrid["feasible"]:
    decomposition_price = (result_classical_hybrid["cost"] / cost_classical - 1) * 100
    print(f"\nPrice of decomposition (classical): +{decomposition_price:.2f}%")

n_ok = sum(1 for r in results if r["status"] == "OK")
print(f"\n{n_ok}/{len(results)} checks passed cleanly.")