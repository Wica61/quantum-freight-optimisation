import json
from pyomo.environ import value

from src.classical_model import build_classical_model
from src.solve import solve_model
from src.hybrid_pipeline import solve_hybrid

FIXTURE = "tests/fixtures/scaled_data_v2.json"
JOINT = 8532.96

# Ratios asymetriques appris par Optuna (diagnostic H3), remis a l'echelle du
# cout maximal de l'instance. Des poids UNIFORMES ne fonctionnent pas : le
# recuit abandonne des envois plutot que de respecter la capacite.
RATIOS = {"assignment": 20.4, "vehicle_activation": 6.3, "emissions": 1.3,
          "hub_activation": 1.2, "capacity": 1.0}


def poids(data):
    mc = max(max(data["transport_cost"].values()),
             max(h["cost"] for h in data["hubs"].values()),
             max(v["fixed_cost"] for v in data["vehicles"].values()))
    return {k: mc * r for k, r in RATIOS.items()}


def test_optimum_joint_inchange():
    """Verrou : si ce chiffre bouge, l'instance ou le modele a change,
    et tous les ecarts du memoire sont a recalculer."""
    data = json.load(open(FIXTURE))
    m = build_classical_model(data)
    solve_model(m, time_limit=300)
    assert abs(value(m.OBJ) - JOINT) < 0.01


def test_sample_coherent():
    """Invariants structurels, et coherence du drapeau feasible : une solution
    declaree faisable DOIT avoir 30 affectations."""
    data = json.load(open(FIXTURE))
    r = solve_hybrid(data, n_clusters=4, penalty_weights=poids(data))
    s = r["sample"]
    assert sum(1 for k in s if k.startswith("y_")) == 3
    assert sum(1 for k in s if k.startswith("u_")) == 16
    if r["feasible"]:
        assert sum(1 for k, v in s.items()
                   if k.startswith("x_") and v == 1) == 30


def test_cout_jamais_sous_optimum():
    """Le garde-fou essentiel : une solution FAISABLE ne peut pas couter moins
    que l'optimum prouve. Si ce test casse, check_feasibility ne verifie pas
    ce qu'on croit."""
    data = json.load(open(FIXTURE))
    for _ in range(3):
        r = solve_hybrid(data, n_clusters=4, penalty_weights=poids(data))
        if r["feasible"]:
            assert r["cost"] >= JOINT - 0.01