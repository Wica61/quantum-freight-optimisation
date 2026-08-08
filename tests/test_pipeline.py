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


def test_sample_complet():
    """867 variables et 30 affectations : detecte une convention de nommage cassee."""
    data = json.load(open(FIXTURE))
    r = solve_hybrid(data, n_clusters=4, penalty_weights=POIDS)
    assert len(r["sample"]) == 867
    assert sum(1 for k, v in r["sample"].items()
               if k.startswith("x_") and v == 1) == 30


def test_cout_jamais_sous_optimum():
    """Le garde-fou essentiel : une solution FAISABLE ne peut pas couter moins
    que l'optimum prouve. Si ce test casse, check_feasibility ne verifie pas
    ce qu'on croit."""
    data = json.load(open(FIXTURE))
    for _ in range(3):
        r = solve_hybrid(data, n_clusters=4, penalty_weights=POIDS)
        if r["feasible"]:
            assert r["cost"] >= JOINT - 0.01