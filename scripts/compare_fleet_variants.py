"""
Compare les variantes de flotte (memes 12 envois, flotte croissante) sur les
trois axes qui comptent : FAISABILITE, QUALITE (ecart a l'optimum) et TEMPS.

Les poids de penalite utilises sont les RAPPORTS appris par Optuna (diagnostic
H3), remis a l'echelle du cout maximal de chaque instance. C'est le point cle :
les poids ne sont pas des constantes mais des rapports relatifs a l'echelle des
couts, ce qui les rend transposables d'une instance a l'autre.

Usage :
    python scripts/compare_fleet_variants.py
    python scripts/compare_fleet_variants.py 2v2t 4v4t
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
from src.hybrid_pipeline import solve_hybrid

RATIOS = {
    "assignment": 20.4,
    "vehicle_activation": 6.3,
    "emissions": 1.3,
    "hub_activation": 1.2,
    "capacity": 1.0,
}

N_RUNS = 5
N_CLUSTERS = 2
NUM_READS = 1000
NUM_SWEEPS = 5000

TAGS = sys.argv[1:] if len(sys.argv) > 1 else ["2v2t", "4v4t", "6v6t"]

rows = []


def scaled_weights(data):
    max_cost = max(
        max(data["transport_cost"].values()),
        max(h["cost"] for h in data["hubs"].values()),
        max(v["fixed_cost"] for v in data["vehicles"].values()),
    )
    return {k: max_cost * r for k, r in RATIOS.items()}, max_cost


for tag in TAGS:
    path = Path(f"tests/fixtures/fixedE_{tag}.json")
    if not path.exists():
        print(f"[ignore] {path} absent -- generez-le d'abord")
        continue

    with open(path) as f:
        data = json.load(f)

    n_veh = len(data["vehicles"])
    cap = sum(v["capacity"] for v in data["vehicles"].values())
    weight = sum(s["weight"] for s in data["shipments"].values())

    print("\n" + "=" * 74)
    print(f"VARIANTE {tag} -- {n_veh} vehicules, capacite {cap}, poids {weight}, "
          f"E_max {data['E_max']}")
    print("=" * 74)

    # ---------- 1. classique joint (deterministe) ----------
    t0 = time.perf_counter()
    model = build_classical_model(data)
    solve_model(model, time_limit=300)
    t_classical = time.perf_counter() - t0
    opt = model.OBJ()
    print(f"[1] Classique joint  : {opt:>9.2f}  en {t_classical:>7.3f}s")
    rows.append({"variante": tag, "methode": "1. classique joint",
                 "vehicules": n_veh, "capacite": cap,
                 "faisable": "1/1", "cout": f"{opt:.2f}", "ecart": "+0.00%",
                 "temps_moyen_s": f"{t_classical:.3f}",
                 "temps_total_s": f"{t_classical:.3f}"})

    w, max_cost = scaled_weights(data)

    # ---------- 2. QUBO joint (stochastique) ----------
    faisables, couts, temps = 0, [], []
    n_vars = None
    t_bloc = time.perf_counter()
    for run in range(N_RUNS):
        t0 = time.perf_counter()
        bqm, _ = build_qubo(data, w)
        best = solve_qubo(bqm, num_reads=NUM_READS, num_sweeps=NUM_SWEEPS)
        dt = time.perf_counter() - t0
        temps.append(dt)
        n_vars = len(bqm.variables)
        if check_feasibility(best.sample, data)["feasible"]:
            faisables += 1
            couts.append(best.energy)
    t_total_joint = time.perf_counter() - t_bloc
    t_moy = statistics.mean(temps)
    if couts:
        ecart = min(couts) / opt - 1
        print(f"[2] QUBO joint       : {min(couts):>9.2f}  en {t_moy:>7.3f}s/run  "
              f"({ecart:+.2%})  faisable {faisables}/{N_RUNS}  vars={n_vars}")
        rows.append({"variante": tag, "methode": "2. QUBO joint",
                     "vehicules": n_veh, "capacite": cap,
                     "faisable": f"{faisables}/{N_RUNS}", "cout": f"{min(couts):.2f}",
                     "ecart": f"{ecart:+.2%}", "temps_moyen_s": f"{t_moy:.3f}",
                     "temps_total_s": f"{t_total_joint:.3f}"})
    else:
        print(f"[2] QUBO joint       :       n/a  en {t_moy:>7.3f}s/run  "
              f"faisable 0/{N_RUNS}  vars={n_vars}")
        rows.append({"variante": tag, "methode": "2. QUBO joint",
                     "vehicules": n_veh, "capacite": cap,
                     "faisable": f"0/{N_RUNS}", "cout": "-", "ecart": "n/a",
                     "temps_moyen_s": f"{t_moy:.3f}",
                     "temps_total_s": f"{t_total_joint:.3f}"})

    # ---------- 3. QUBO decompose (stochastique) ----------
    faisables, couts, temps = 0, [], []
    t_bloc = time.perf_counter()
    for run in range(N_RUNS):
        t0 = time.perf_counter()
        r = solve_hybrid(data, n_clusters=N_CLUSTERS, penalty_weights=w)
        temps.append(time.perf_counter() - t0)
        if r["feasible"]:
            faisables += 1
            couts.append(r["cost"])
    t_total_dec = time.perf_counter() - t_bloc
    t_moy = statistics.mean(temps)
    if couts:
        ecart = min(couts) / opt - 1
        print(f"[3] QUBO decompose   : {min(couts):>9.2f}  en {t_moy:>7.3f}s/run  "
              f"({ecart:+.2%})  faisable {faisables}/{N_RUNS}")
        rows.append({"variante": tag, "methode": "3. QUBO decompose",
                     "vehicules": n_veh, "capacite": cap,
                     "faisable": f"{faisables}/{N_RUNS}", "cout": f"{min(couts):.2f}",
                     "ecart": f"{ecart:+.2%}", "temps_moyen_s": f"{t_moy:.3f}",
                     "temps_total_s": f"{t_total_dec:.3f}"})
    else:
        print(f"[3] QUBO decompose   :       n/a  en {t_moy:>7.3f}s/run  "
              f"faisable 0/{N_RUNS}")
        rows.append({"variante": tag, "methode": "3. QUBO decompose",
                     "vehicules": n_veh, "capacite": cap,
                     "faisable": f"0/{N_RUNS}", "cout": "-", "ecart": "n/a",
                     "temps_moyen_s": f"{t_moy:.3f}",
                     "temps_total_s": f"{t_total_dec:.3f}"})


# ---------------------------------------------------------------- tableau
print("\n\n" + "=" * 74)
print("TABLEAU RECAPITULATIF")
print("=" * 74)

headers = ["Variante", "Methode", "Veh.", "Cap.", "Faisable", "Cout",
           "Ecart", "Temps/run", "Temps total"]
keys = ["variante", "methode", "vehicules", "capacite", "faisable", "cout",
        "ecart", "temps_moyen_s", "temps_total_s"]

widths = [max(len(h), max((len(str(r[k])) for r in rows), default=0))
          for h, k in zip(headers, keys)] if rows else []

if rows:
    sep = "  "
    print(sep.join(h.ljust(wd) for h, wd in zip(headers, widths)))
    print(sep.join("-" * wd for wd in widths))
    current = None
    for r in rows:
        if current is not None and r["variante"] != current:
            print(sep.join("-" * wd for wd in widths))
        current = r["variante"]
        print(sep.join(str(r[k]).ljust(wd) for k, wd in zip(keys, widths)))

    out = Path("results")
    out.mkdir(exist_ok=True)
    csv_path = out / "fleet_variants_timed.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nEnregistre dans {csv_path}")

    print("\nNotes de lecture :")
    print("  - 'Temps/run' = temps moyen d'UNE execution (construction du QUBO incluse).")
    print("  - 'Temps total' = temps du bloc complet des 5 executions.")
    print("  - Le classique est deterministe : 1 execution suffit, d'ou temps/run = total.")
    print("  - L'ecart n'est calcule que sur les executions FAISABLES.")
    print("  - L'optimum classique differe legerement d'une variante a l'autre :")
    print("    E_max est recalibre pour chaque flotte, les instances ne sont donc")
    print("    pas rigoureusement identiques -- seuls les 12 envois le sont.")