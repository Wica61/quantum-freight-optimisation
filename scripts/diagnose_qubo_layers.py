"""
DIAGNOSTIC EN ESCALIER -- localise la couche du QUBO qui casse la faisabilite.

  Etage 1 : affectation seule                  -> le plus simple possible
  Etage 2 : + activation hub/vehicule
  Etage 3 : + couts reels (objectif)
  Etage 4 : QUBO complet (via build_qubo)      -> + capacite + emissions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import itertools
import dimod

from src.qubo_model import build_qubo
from src.solve_qubo import solve_qubo
from src.feasibility import check_feasibility

DATA_PATH = "tests/fixtures/mid_data.json"
P = 500.0
NUM_READS = 1000
NUM_SWEEPS = 5000

with open(DATA_PATH) as f:
    data = json.load(f)


def xvars_of(i):
    out = []
    for hv in data["valid_combinations"][i]:
        h, v = hv.split("|")
        out.append((f"x_{i}_{h}_{v}", h, v))
    return out


def add_assignment(bqm):
    """P*(somme_x - 1)^2 developpe a la main (x^2 = x en binaire)."""
    for i in data["shipments"]:
        names = [n for n, _, _ in xvars_of(i)]
        for n in names:
            bqm.add_linear(n, -P)
        for a, b in itertools.combinations(names, 2):
            bqm.add_quadratic(a, b, 2 * P)


def add_activation(bqm):
    """P*x*(1-y) + P*x*(1-u) = 2P*x - P*x*y - P*x*u."""
    for h in data["hubs"]:
        bqm.add_variable(f"y_{h}", 0.0)
    for v in data["vehicles"]:
        bqm.add_variable(f"u_{v}", 0.0)
    for i in data["shipments"]:
        for n, h, v in xvars_of(i):
            bqm.add_linear(n, 2 * P)
            bqm.add_quadratic(n, f"y_{h}", -P)
            bqm.add_quadratic(n, f"u_{v}", -P)


def add_objective(bqm):
    for i in data["shipments"]:
        for n, h, v in xvars_of(i):
            bqm.add_linear(n, data["transport_cost"][f"{i}|{h}|{v}"])
    for h in data["hubs"]:
        bqm.add_linear(f"y_{h}", data["hubs"][h]["cost"])
    for v in data["vehicles"]:
        bqm.add_linear(f"u_{v}", data["vehicles"][v]["fixed_cost"])


def bad_assignment(sample):
    bad = []
    for i in data["shipments"]:
        n = sum(sample.get(name, 0) for name, _, _ in xvars_of(i))
        if n != 1:
            bad.append(f"{i}:{n}")
    return bad


def bad_activation(sample):
    bad = []
    for i in data["shipments"]:
        for name, h, v in xvars_of(i):
            if sample.get(name, 0) == 1:
                if sample.get(f"y_{h}", 0) != 1:
                    bad.append(f"{name}->y_{h}=0")
                if sample.get(f"u_{v}", 0) != 1:
                    bad.append(f"{name}->u_{v}=0")
    return bad


def run_level(label, bqm, checks):
    best = solve_qubo(bqm, num_reads=NUM_READS, num_sweeps=NUM_SWEEPS)
    print(f"\n--- {label} ---")
    print(f"  variables dans le BQM : {len(bqm.variables)}")
    print(f"  energie               : {best.energy:.2f}")
    total = 0
    for cname, fn in checks:
        issues = fn(best.sample)
        total += len(issues)
        status = "OK" if not issues else f"{len(issues)} PROBLEME(S)"
        print(f"  {cname:<22} : {status}")
        if issues:
            print(f"      {issues[:8]}")
    print(f"  => {'FAISABLE a cet etage' if total == 0 else 'ECHEC a cet etage'}")
    return total == 0


print("=" * 70)
print(f"DIAGNOSTIC EN ESCALIER -- {DATA_PATH}")
print(f"envois={len(data['shipments'])}  hubs={len(data['hubs'])}  "
      f"vehicules={len(data['vehicles'])}  P={P}")
print("=" * 70)

bqm1 = dimod.BinaryQuadraticModel(vartype="BINARY")
for i in data["shipments"]:
    for n, _, _ in xvars_of(i):
        bqm1.add_variable(n, 0.0)
add_assignment(bqm1)
ok1 = run_level("ETAGE 1 : affectation seule", bqm1,
                [("affectation", bad_assignment)])

bqm2 = dimod.BinaryQuadraticModel(vartype="BINARY")
for i in data["shipments"]:
    for n, _, _ in xvars_of(i):
        bqm2.add_variable(n, 0.0)
add_assignment(bqm2)
add_activation(bqm2)
ok2 = run_level("ETAGE 2 : affectation + activation", bqm2,
                [("affectation", bad_assignment), ("activation", bad_activation)])

bqm3 = dimod.BinaryQuadraticModel(vartype="BINARY")
for i in data["shipments"]:
    for n, _, _ in xvars_of(i):
        bqm3.add_variable(n, 0.0)
add_assignment(bqm3)
add_activation(bqm3)
add_objective(bqm3)
ok3 = run_level("ETAGE 3 : + couts reels (objectif)", bqm3,
                [("affectation", bad_assignment), ("activation", bad_activation)])

weights = {"assignment": P, "hub_activation": P, "vehicle_activation": P,
           "capacity": P, "emissions": P}
bqm4, _ = build_qubo(data, weights)
best4 = solve_qubo(bqm4, num_reads=NUM_READS, num_sweeps=NUM_SWEEPS)
check4 = check_feasibility(best4.sample, data)
print("\n--- ETAGE 4 : QUBO complet (build_qubo) ---")
print(f"  variables dans le BQM : {len(bqm4.variables)}")
print(f"  energie               : {best4.energy:.2f}")
print(f"  faisable              : {check4['feasible']}")
print(f"  violations            : {len(check4['issues'])}")
for issue in check4["issues"]:
    print(f"      - {issue}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
if not ok1:
    print("L'ETAGE 1 echoue : l'encodage de la contrainte d'egalite est en cause.")
    print("C'est le cas le plus grave -- aucun poids ne peut le corriger.")
elif not ok2:
    print("L'ETAGE 2 echoue : les termes d'activation hub/vehicule sont en cause.")
elif not ok3:
    print("L'ETAGE 3 echoue : l'objectif ECRASE les penalites.")
    print("Les couts reels sont trop grands devant P -- il faut augmenter P,")
    print("ou normaliser l'objectif (le diviser par son ordre de grandeur).")
elif not check4["feasible"]:
    print("Seul l'ETAGE 4 echoue : la capacite ou les emissions (encodage par")
    print("variables d'ecart) sont en cause. Regarder la liste des violations")
    print("ci-dessus pour savoir laquelle des deux.")
else:
    print("Tous les etages passent -- le probleme serait alors purement")
    print("stochastique, a re-tester sur plusieurs executions.")