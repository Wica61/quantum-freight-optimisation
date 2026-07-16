# src/solve.py
from pyomo.environ import SolverFactory, TerminationCondition
from src.config import GUROBI_OPTIONS

def solve_model(model, time_limit=60, mip_gap=0.0):
    solver_options = dict(GUROBI_OPTIONS)
    solver_options["TimeLimit"] = time_limit
    solver_options["MIPGap"] = mip_gap

    solver = SolverFactory('gurobi_direct', manage_env=True, options=solver_options)
    result = solver.solve(model, tee=True)
    status = result.solver.termination_condition

    if status == TerminationCondition.optimal:
        print(f"✓ Solution optimale trouvée. Coût total : {model.OBJ():,.2f}")
    elif status == TerminationCondition.infeasible:
        print("✗ Modèle infaisable.")
    else:
        print(f"⚠ Statut du solveur : {status}")

    solver.close()
    return result

"""
python -c "
import json
from src.classical_model import build_classical_model
from src.solve import solve_model

with open('tests/fixtures/toy_data.json') as f:
    data = json.load(f)

model = build_classical_model(data)
solve_model(model)
"
"""