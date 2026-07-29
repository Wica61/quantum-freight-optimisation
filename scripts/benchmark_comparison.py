"""
Benchmark comparatif -- compare TOUTES les methodes sur LE MEME jeu de donnees.

Methodes comparees :
  1. Classique joint       (Gurobi, exact)            -- reference / optimum
  2. Classique decompose   (Gurobi par cluster)       -- mesure le prix de la decomposition
  3. QUBO joint            (recuit simule, sans decomposition)
  4. QUBO decompose        (recuit simule par cluster)

Methodologie :
  - Les methodes 1 et 2 sont DETERMINISTES (Gurobi + KMeans a graine fixe) : 1 execution.
  - Les methodes 3 et 4 sont STOCHASTIQUES (recuit simule) : N executions, et on
    rapporte un TAUX DE FAISABILITE + le meilleur / le pire resultat, jamais un
    seul tirage (qui ne prouverait rien).
  - Le "gap" n'est calcule QUE sur les solutions faisables : comparer le cout d'une
    solution qui viole les contraintes a celui d'une solution valide n'a pas de sens
    (une solution invalide peut etre "moins chere" precisement PARCE QU'elle triche).

Sortie : un tableau lisible + un CSV reutilisable directement dans le rapport.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import csv
import statistics

from src.classical_model import build_classical_model
from src.solve import solve_model
from src.qubo_model import build_qubo
from src.solve_qubo import solve_qubo
from src.feasibility import check_feasibility
from src.decomposition_agent import decompose_network
from src.hybrid_pipeline import solve_hybrid, solve_hybrid_classical

# ---------------------------------------------------------------- parametres
N_RUNS_STOCHASTIC = 5     # repetitions pour les methodes a recuit simule
N_CLUSTERS = 2
NUM_READS = 1000
MANUAL_WEIGHTS = {"assignment": 500, "hub_activation": 500, "vehicle_activation": 500,
                  "capacity": 500, "emissions": 500}

DATASETS = [
    ("Toy (5 shipments)", "tests/fixtures/toy_data.json"),
    ("Mid (12 shipments)", "tests/fixtures/mid_data.json"),
    ("Real (30 shipments)", "tests/fixtures/scaled_data.json"),
]

all_rows = []


# ---------------------------------------------------------------- utilitaires
def real_cost_from_sample(sample, data):
    """Recalcule le cout METIER reel d'un echantillon (hors penalites QUBO).
    Indispensable : l'energie QUBO melange cout reel et penalites, elle n'est
    donc PAS comparable au cout d'une solution classique."""
    cost = 0.0
    for k, v in sample.items():
        if v != 1:
            continue
        if k.startswith("y_"):
            h = k.split("_", 1)[1]
            if h in data["hubs"]:
                cost += data["hubs"][h]["cost"]
        elif k.startswith("u_"):
            veh = k.split("_", 1)[1]
            if veh in data["vehicles"]:
                cost += data["vehicles"][veh]["fixed_cost"]
        elif k.startswith("x_"):
            parts = k.split("_", 3)
            if len(parts) == 4:
                _, i, h, veh = parts
                key = f"{i}|{h}|{veh}"
                if key in data["transport_cost"]:
                    cost += data["transport_cost"][key]
    return cost


def add_row(dataset, method, feasible_runs, total_runs, best_cost, worst_cost,
            mean_time, optimum, note=""):
    """Enregistre une ligne du tableau. Le gap n'est renseigne que si au moins
    une execution a produit une solution faisable."""
    if feasible_runs > 0 and best_cost is not None and optimum:
        gap = (best_cost / optimum - 1) * 100
        gap_str = f"{gap:+.2f}%"
    else:
        gap_str = "n/a"
    all_rows.append({
        "dataset": dataset,
        "method": method,
        "feasible": f"{feasible_runs}/{total_runs}",
        "best_cost": f"{best_cost:.2f}" if best_cost is not None else "-",
        "worst_cost": f"{worst_cost:.2f}" if worst_cost is not None else "-",
        "gap_vs_optimum": gap_str,
        "avg_time_s": f"{mean_time:.2f}",
        "note": note,
    })


# ---------------------------------------------------------------- benchmark
for dataset_name, path in DATASETS:
    print("\n" + "=" * 78)
    print(f"BENCHMARK -- {dataset_name}")
    print("=" * 78)

    with open(path) as f:
        data = json.load(f)
    print(f"E_max = {data['E_max']}   |   shipments = {len(data['shipments'])}   "
          f"|   hubs = {len(data['hubs'])}   |   vehicles = {len(data['vehicles'])}")

    # ---- Methode 1 : classique joint (exact, deterministe) ----
    print("\n[1/4] Classical joint (Gurobi, exact)...")
    t0 = time.time()
    model = build_classical_model(data)
    solve_model(model, time_limit=300)
    t_classical = time.time() - t0
    optimum = model.OBJ()
    add_row(dataset_name, "1. Classical joint", 1, 1, optimum, optimum,
            t_classical, optimum, "exact reference")
    print(f"      optimum = {optimum:.2f}  [{t_classical:.2f}s]")

    # ---- Methode 2 : classique decompose (deterministe) ----
    print("\n[2/4] Classical decomposed (Gurobi per cluster)...")
    t0 = time.time()
    res = solve_hybrid_classical(data, n_clusters=N_CLUSTERS)
    t_cd = time.time() - t0
    if res["cost"] is None:
        # un cluster est infaisable isolement : la decomposition ne tient pas
        add_row(dataset_name, "2. Classical decomposed", 0, 1, None, None,
                t_cd, optimum, "decomposition NOT viable at this scale")
        print(f"      INFEASIBLE -- {res['issues'][0]}")
    else:
        feas = 1 if res["feasible"] else 0
        add_row(dataset_name, "2. Classical decomposed", feas, 1,
                res["cost"], res["cost"], t_cd, optimum,
                "price of decomposition" if feas else f"{len(res['issues'])} violations")
        print(f"      cost = {res['cost']:.2f}  feasible = {res['feasible']}  [{t_cd:.2f}s]")

    # ---- Methode 3 : QUBO joint (stochastique) ----
    print(f"\n[3/4] QUBO joint (simulated annealing, {N_RUNS_STOCHASTIC} runs)...")
    bqm, _ = build_qubo(data, MANUAL_WEIGHTS)
    feasible_costs, all_costs, times, issue_counts = [], [], [], []
    for r in range(N_RUNS_STOCHASTIC):
        t0 = time.time()
        best = solve_qubo(bqm, num_reads=NUM_READS)
        times.append(time.time() - t0)
        check = check_feasibility(best.sample, data)
        cost = real_cost_from_sample(best.sample, data)
        all_costs.append(cost)
        issue_counts.append(len(check["issues"]))
        if check["feasible"]:
            feasible_costs.append(cost)
        print(f"      run {r+1}: cost={cost:.2f}  feasible={check['feasible']}  "
              f"violations={len(check['issues'])}")
    if feasible_costs:
        add_row(dataset_name, "3. QUBO joint", len(feasible_costs), N_RUNS_STOCHASTIC,
                min(feasible_costs), max(feasible_costs), statistics.mean(times), optimum,
                "best/worst among feasible runs")
    else:
        add_row(dataset_name, "3. QUBO joint", 0, N_RUNS_STOCHASTIC,
                min(all_costs), max(all_costs), statistics.mean(times), optimum,
                f"NO feasible run; {min(issue_counts)}-{max(issue_counts)} violations "
                f"(costs not comparable)")

    # ---- Methode 4 : QUBO decompose (stochastique) ----
    print(f"\n[4/4] QUBO decomposed (simulated annealing per cluster, "
          f"{N_RUNS_STOCHASTIC} runs)...")
    feasible_costs, all_costs, times, issue_counts = [], [], [], []
    for r in range(N_RUNS_STOCHASTIC):
        t0 = time.time()
        res = solve_hybrid(data, n_clusters=N_CLUSTERS, penalty_weights=MANUAL_WEIGHTS)
        times.append(time.time() - t0)
        all_costs.append(res["cost"])
        issue_counts.append(len(res.get("issues", [])))
        if res["feasible"]:
            feasible_costs.append(res["cost"])
        print(f"      run {r+1}: cost={res['cost']:.2f}  feasible={res['feasible']}  "
              f"violations={len(res.get('issues', []))}")
    if feasible_costs:
        add_row(dataset_name, "4. QUBO decomposed", len(feasible_costs), N_RUNS_STOCHASTIC,
                min(feasible_costs), max(feasible_costs), statistics.mean(times), optimum,
                "best/worst among feasible runs")
    else:
        add_row(dataset_name, "4. QUBO decomposed", 0, N_RUNS_STOCHASTIC,
                min(all_costs), max(all_costs), statistics.mean(times), optimum,
                f"NO feasible run; {min(issue_counts)}-{max(issue_counts)} violations "
                f"(costs not comparable)")


# ---------------------------------------------------------------- tableau final
print("\n\n" + "=" * 78)
print("COMPARISON TABLE")
print("=" * 78)

headers = ["Dataset", "Method", "Feasible", "Best cost", "Worst cost",
           "Gap vs opt.", "Avg time", "Note"]
keys = ["dataset", "method", "feasible", "best_cost", "worst_cost",
        "gap_vs_optimum", "avg_time_s", "note"]

widths = []
for h, k in zip(headers, keys):
    w = max(len(h), max((len(str(row[k])) for row in all_rows), default=0))
    widths.append(w)

sep = "  "
print(sep.join(h.ljust(w) for h, w in zip(headers, widths)))
print(sep.join("-" * w for w in widths))
current_dataset = None
for row in all_rows:
    if current_dataset is not None and row["dataset"] != current_dataset:
        print(sep.join("-" * w for w in widths))
    current_dataset = row["dataset"]
    print(sep.join(str(row[k]).ljust(w) for k, w in zip(keys, widths)))

# ---------------------------------------------------------------- export CSV
out = Path("results")
out.mkdir(exist_ok=True)
csv_path = out / "benchmark_comparison.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\nSaved to {csv_path}")
print("\nReading notes:")
print("  - 'Feasible' = how many runs produced a solution respecting ALL constraints.")
print("  - Gap is computed on the BEST FEASIBLE run only; 'n/a' means no run was feasible.")
print("  - A low cost with 0 feasible runs is NOT a good result: an invalid solution can")
print("    look cheap precisely because it breaks the rules it was supposed to respect.")
print("  - Deterministic methods (1, 2) show 1/1 by design, not by luck.")