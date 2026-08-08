import json
from pyomo.environ import value

from src.classical_model import build_classical_model
from src.solve import solve_model
from src.hybrid_pipeline import solve_hybrid

FIXTURE = "tests/fixtures/scaled_data_v2.json"
JOINT = 8532.96
POIDS = {k: 12000 for k in ["assignment", "hub_activation",
                            "vehicle_activation", "capacity", "emissions"]}


def test_optimum_joint_inchange():
    """Verrou : si ce chiffre bouge, l'instance ou le modele a change,
    et tous les ecarts du memoire sont a recalculer."""
    data = json.load(open(FIXTURE))
    m = build_classical_model(data)
    solve_model(m, time_limit=300)
    assert abs(value(m.OBJ) - JOINT) < 0.01


def test_sample_coherent():
    """Invariants structurels toujours vrais, et coherence du drapeau feasible :
    une solution declaree faisable DOIT avoir 30 affectations. Le recuit etant
    stochastique, un run infaisable n'est pas un bug -- mais un run faisable
    avec un compte different le serait."""
    data = json.load(open(FIXTURE))
    r = solve_hybrid(data, n_clusters=4, penalty_weights=POIDS)
    s = r["sample"]
    assert sum(1 for k in s if k.startswith("y_")) == 3
    assert sum(1 for k in s if k.startswith("u_")) == 16
    n_affectations = sum(1 for k, v in s.items() if k.startswith("x_") and v == 1)
    if r["feasible"]:
        assert n_affectations == 30


def test_cout_jamais_sous_optimum():
    """Le garde-fou essentiel : une solution FAISABLE ne peut pas couter moins
    que l'optimum prouve. Si ce test casse, check_feasibility ne verifie pas
    ce qu'on croit."""
    data = json.load(open(FIXTURE))
    for _ in range(3):
        r = solve_hybrid(data, n_clusters=4, penalty_weights=POIDS)
        if r["feasible"]:
            assert r["cost"] >= JOINT - 0.01