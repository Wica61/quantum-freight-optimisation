"""
Tests des hypotheses restantes sur l'echec de faisabilite du QUBO.

Usage :
    python scripts/diagnose_hypotheses.py H5    # commencer par celle-ci
    python scripts/diagnose_hypotheses.py H2
    python scripts/diagnose_hypotheses.py H8
    python scripts/diagnose_hypotheses.py H6
    python scripts/diagnose_hypotheses.py H3    # la plus longue (~6 min)
    python scripts/diagnose_hypotheses.py all
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time

from dwave.samplers import SimulatedAnnealingSampler
from src.classical_model import build_classical_model
from src.solve import solve_model
from src.qubo_model import build_qubo
from src.solve_qubo import solve_qubo
from src.feasibility import check_feasibility

DATA_PATH = "tests/fixtures/mid_data.json"
N_RUNS = 5

with open(DATA_PATH) as f:
    data = json.load(f)

MAX_COST = max(
    max(data["transport_cost"].values()),
    max(h["cost"] for h in data["hubs"].values()),
    max(v["fixed_cost"] for v in data["vehicles"].values()),
)
P_REF = MAX_COST * 10
UNIFORM = {k: P_REF for k in ["assignment", "hub_activation",
                              "vehicle_activation", "capacity", "emissions"]}


def classical_optimum():
    model = build_classical_model(data)
    solve_model(model, time_limit=120)
    return model.OBJ()


# =====================================================================
# H5 -- LE TEST DECISIF : le QUBO est-il correctement encode ?
# =====================================================================
def test_h5(opt):
    """Compare l'energie minimale trouvee par le recuit au COUT REEL de
    l'optimum classique. Si une solution est faisable, toutes les penalites
    valent zero, donc son energie DOIT egaler son cout reel.

      - energie trouvee < cout optimal -> un etat infaisable est insuffisamment
        penalise. ENCODAGE FAUTIF.
      - energie trouvee > cout optimal, sans faisabilite -> l'encodage est
        correct, le recuit ne l'ATTEINT pas. PROBLEME DE RECHERCHE."""
    print(f"\nCout de l'optimum classique (faisable, prouve) : {opt:.2f}")
    print("Une solution faisable a une energie EGALE a son cout reel\n"
          "(toutes les penalites sont nulles).\n")

    energies = []
    for run in range(N_RUNS):
        bqm, _ = build_qubo(data, UNIFORM)
        best = solve_qubo(bqm, num_reads=1000, num_sweeps=5000)
        check = check_feasibility(best.sample, data)
        energies.append(best.energy)
        flag = ""
        if best.energy < opt - 0.01:
            flag = "  <-- INFERIEURE a l'optimum : anomalie"
        print(f"  run {run+1}: energie={best.energy:>12.2f}  "
              f"faisable={str(check['feasible']):<5}  "
              f"violations={len(check['issues'])}{flag}")

    e_min = min(energies)
    print()
    if e_min < opt - 0.01:
        print("=> VERDICT : ENCODAGE FAUTIF.")
        print(f"   Le recuit atteint {e_min:.2f}, en dessous du cout optimal reel")
        print(f"   ({opt:.2f}). Un etat infaisable est donc insuffisamment penalise.")
        print("   Aucune augmentation de puissance de calcul ne corrigera cela :")
        print("   il faut identifier la contrainte sous-penalisee (voir H8).")
    else:
        print("=> VERDICT : ENCODAGE CORRECT.")
        print(f"   Aucune energie ({e_min:.2f}) ne descend sous le cout optimal")
        print(f"   ({opt:.2f}). Le minimum du QUBO est bien la bonne solution ;")
        print("   c'est le recuit qui ne l'atteint pas. Le probleme est la")
        print("   RECHERCHE -> voir H2 (sweeps), H3 (asymetrie), H6 (temperature).")


# =====================================================================
# H2 -- num_sweeps plutot que num_reads
# =====================================================================
def test_h2(opt):
    """num_reads = tentatives independantes (deja elimine comme levier).
    num_sweeps = duree du recuit A L'INTERIEUR de chaque tentative."""
    print("\nnum_reads fixe a 200 ; num_sweeps varie (budget compare, pas egal)\n")
    for sweeps in [1000, 5000, 20000, 100000]:
        faisables, viol_min, t0 = 0, 99, time.time()
        for run in range(N_RUNS):
            bqm, _ = build_qubo(data, UNIFORM)
            best = solve_qubo(bqm, num_reads=200, num_sweeps=sweeps)
            check = check_feasibility(best.sample, data)
            viol_min = min(viol_min, len(check["issues"]))
            if check["feasible"]:
                faisables += 1
        dt = (time.time() - t0) / N_RUNS
        print(f"  num_sweeps={sweeps:>7}  faisable {faisables}/{N_RUNS}  "
              f"meilleur={viol_min} violation(s)  [{dt:.1f}s/run]")
    print("\n=> Si le taux monte franchement avec num_sweeps a budget bien")
    print("   inferieur a celui de num_reads=20000, alors c'etait le bon levier")
    print("   depuis le debut (chaque tentative doit avoir le temps de converger,")
    print("   multiplier les tentatives trop courtes ne sert a rien).")


# =====================================================================
# H8 -- defaut residuel dans l'encodage de la capacite
# =====================================================================
def test_h8(opt):
    """Isolation refaite avec P correctement calibre, en regardant le TYPE de
    violation -- pas seulement le nombre."""
    print(f"\nP calibre a {P_REF:.0f} (10x le cout maximal) pour chaque test\n")
    base = {"assignment": P_REF, "hub_activation": P_REF, "vehicle_activation": P_REF}
    configs = [
        ("sans capacite ni emissions", {**base, "capacity": 0, "emissions": 0}),
        ("capacite seule", {**base, "capacity": P_REF, "emissions": 0}),
        ("emissions seules", {**base, "capacity": 0, "emissions": P_REF}),
        ("les deux", {**base, "capacity": P_REF, "emissions": P_REF}),
    ]
    for label, w in configs:
        types = {"affectation": 0, "activation": 0, "capacite": 0, "emissions": 0}
        faisables = 0
        for run in range(N_RUNS):
            bqm, _ = build_qubo(data, w)
            best = solve_qubo(bqm, num_reads=1000, num_sweeps=5000)
            check = check_feasibility(best.sample, data)
            if check["feasible"]:
                faisables += 1
            for issue in check["issues"]:
                if "affectation" in issue:
                    types["affectation"] += 1
                elif "hub non active" in issue or "vehicule non active" in issue:
                    types["activation"] += 1
                elif "charge" in issue:
                    types["capacite"] += 1
                elif "emissions" in issue:
                    types["emissions"] += 1
        detail = "  ".join(f"{k}={v}" for k, v in types.items() if v)
        print(f"  {label:<28} faisable {faisables}/{N_RUNS}   {detail or 'aucune violation'}")
    print("\n=> Si des violations de type 'capacite' apparaissent alors que la")
    print("   capacite est ACTIVE et fortement penalisee, l'encodage de cette")
    print("   contrainte est en cause. Si les violations restent de type")
    print("   'affectation', c'est le compromis global qui est en jeu, pas la")
    print("   capacite en elle-meme.")


# =====================================================================
# H6 -- calendrier de temperature (beta_range)
# =====================================================================
def test_h6(opt):
    """dimod calcule beta_range automatiquement a partir de l'amplitude des
    biais. Avec des biais de ~30 (couts) a ~9000 (penalites), ce calcul peut
    produire un calendrier inadapte.
    beta = temperature inverse : beta faible = chaud, beta eleve = froid."""
    print("\nbeta_range explicite, num_reads=1000, num_sweeps=5000\n")
    sampler = SimulatedAnnealingSampler()
    bqm, _ = build_qubo(data, UNIFORM)

    ranges = [None, [0.0001, 1.0], [0.001, 10.0], [0.01, 100.0], [0.1, 1000.0]]
    for br in ranges:
        faisables, viol_min = 0, 99
        for run in range(N_RUNS):
            kwargs = dict(num_reads=1000, num_sweeps=5000)
            if br is not None:
                kwargs["beta_range"] = br
            ss = sampler.sample(bqm, **kwargs)
            best = ss.first
            check = check_feasibility(best.sample, data)
            viol_min = min(viol_min, len(check["issues"]))
            if check["feasible"]:
                faisables += 1
        label = "auto (defaut dimod)" if br is None else f"{br}"
        print(f"  beta_range={label:<22} faisable {faisables}/{N_RUNS}  "
              f"meilleur={viol_min} violation(s)")
    print("\n=> Si une plage explicite fait nettement mieux que le calcul")
    print("   automatique, le calendrier de temperature etait le goulot")
    print("   d'etranglement -- resultat interessant et peu documente.")


# =====================================================================
# H3 -- poids asymetriques (Optuna sur une plage derivee de l'echelle)
# =====================================================================
def test_h3(opt):
    """Les 30 essais precedents cherchaient dans une plage fixe 50-20000 AVEC
    le bug EMISSION_SCALE actif -- tout point y etait infaisable. On relance
    avec le bug corrige, une plage derivee de l'echelle, un essai amorce, et
    une evaluation moins bruitee."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    lo, hi = MAX_COST, MAX_COST * 50
    print(f"\nPlage de recherche : {lo:.0f} a {hi:.0f} (derivee de max_cost={MAX_COST:.0f})")
    print(f"Essai 0 amorce sur la valeur uniforme P={P_REF:.0f}\n")

    keys = ["assignment", "hub_activation", "vehicle_activation", "capacity", "emissions"]

    def objective(trial):
        w = {k: trial.suggest_float(k, lo, hi, log=True) for k in keys}
        best_energy = None
        for _ in range(2):
            bqm, _ = build_qubo(data, w)
            best = solve_qubo(bqm, num_reads=1000, num_sweeps=5000)
            if check_feasibility(best.sample, data)["feasible"]:
                if best_energy is None or best.energy < best_energy:
                    best_energy = best.energy
        return best_energy if best_energy is not None else 1e6

    study = optuna.create_study(direction="minimize")
    study.enqueue_trial({k: P_REF for k in keys})
    n_trials = 25
    study.optimize(objective, n_trials=n_trials)

    feasible_trials = [t for t in study.trials
                       if t.value is not None and t.value < 1e6]
    print(f"Essais faisables : {len(feasible_trials)}/{n_trials}")
    if feasible_trials:
        bt = min(feasible_trials, key=lambda t: t.value)
        print(f"Meilleure energie : {bt.value:.2f}  (optimum classique {opt:.2f}, "
              f"ecart {bt.value/opt-1:+.2%})")
        print("Poids correspondants :")
        for k, v in bt.params.items():
            print(f"   {k:<20} {v:>10.1f}")
        ratios = {k: v / min(bt.params.values()) for k, v in bt.params.items()}
        print("Rapports entre poids (normalises sur le plus petit) :")
        for k, r in sorted(ratios.items(), key=lambda kv: -kv[1]):
            print(f"   {k:<20} x{r:>6.1f}")
        print("\n=> Si ces rapports sont tres inegaux, l'ASYMETRIE etait bien")
        print("   necessaire, et les poids uniformes ne pouvaient pas marcher.")
    else:
        print("\n=> Aucun essai faisable meme avec une plage correcte et le bug")
        print("   corrige : l'asymetrie des poids n'est PAS le levier manquant.")
        print("   Reste alors H5 (encodage) et H6 (temperature) comme causes.")


# =====================================================================
if __name__ == "__main__":
    which = sys.argv[1].upper() if len(sys.argv) > 1 else "H5"
    print("=" * 72)
    print(f"DIAGNOSTIC {which} -- {DATA_PATH}")
    print(f"envois={len(data['shipments'])}  vehicules={len(data['vehicles'])}  "
          f"max_cost={MAX_COST:.2f}  P_ref={P_REF:.0f}")
    print("=" * 72)

    opt = classical_optimum()

    tests = {"H2": test_h2, "H3": test_h3, "H5": test_h5,
             "H6": test_h6, "H8": test_h8}
    if which == "ALL":
        for name in ["H5", "H8", "H2", "H6", "H3"]:
            print("\n" + "#" * 72)
            print(f"# {name}")
            print("#" * 72)
            tests[name](opt)
    elif which in tests:
        tests[which](opt)
    else:
        print(f"Inconnu : {which}. Choix : H2, H3, H5, H6, H8, all")