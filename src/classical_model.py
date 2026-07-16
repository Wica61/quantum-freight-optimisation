# src/classical_model.py
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    Binary, minimize
)

def build_classical_model(data):
    m = ConcreteModel()
    m.I = Set(initialize=list(data["shipments"].keys()))
    m.H = Set(initialize=list(data["hubs"].keys()))
    m.V = Set(initialize=list(data["vehicles"].keys()))

    # Une seule variable x par combinaison VALIDE (fenetre de temps deja filtree
    # a la generation des donnees, etape 3) -- pas de variable pour les combinaisons
    # hors fenetre, donc pas besoin de contrainte de fenetre de temps separee.
    valid_pairs = [
        (i, h, v)
        for i in data["shipments"]
        for hv in data["valid_combinations"][i]
        for h, v in [hv.split("|")]
    ]
    m.VALID = Set(initialize=valid_pairs, dimen=3)

    m.y = Var(m.H, domain=Binary)             # hub ouvert
    m.x = Var(m.VALID, domain=Binary)          # affectation envoi -> hub -> vehicule
    m.u = Var(m.V, domain=Binary)              # vehicule utilise

    def obj_rule(mdl):
        return (
            sum(data["hubs"][h]["cost"] * mdl.y[h] for h in mdl.H)
            + sum(data["transport_cost"][f"{i}|{h}|{v}"] * mdl.x[i, h, v] for (i, h, v) in mdl.VALID)
            + sum(data["vehicles"][v]["fixed_cost"] * mdl.u[v] for v in mdl.V)
        )
    m.OBJ = Objective(rule=obj_rule, sense=minimize)

    def assignment_rule(mdl, i):
        pairs = [(h, v) for (ii, h, v) in mdl.VALID if ii == i]
        return sum(mdl.x[i, h, v] for h, v in pairs) == 1
    m.assignment = Constraint(m.I, rule=assignment_rule)

    def hub_activation_rule(mdl, i, h, v):
        return mdl.x[i, h, v] <= mdl.y[h]
    m.hub_activation = Constraint(m.VALID, rule=hub_activation_rule)

    def vehicle_activation_rule(mdl, i, h, v):
        return mdl.x[i, h, v] <= mdl.u[v]
    m.vehicle_activation = Constraint(m.VALID, rule=vehicle_activation_rule)

    def capacity_rule(mdl, v):
        pairs = [(i, h) for (i, h, vv) in mdl.VALID if vv == v]
        return sum(data["shipments"][i]["weight"] * mdl.x[i, h, v] for i, h in pairs) <= data["vehicles"][v]["capacity"] * mdl.u[v]
    m.capacity = Constraint(m.V, rule=capacity_rule)
    
    def emissions_cap_rule(mdl):
        return sum(data["emission"][f"{i}|{h}|{v}"] * mdl.x[i, h, v] for (i, h, v) in mdl.VALID) <= data["E_max"]
    m.emissions_cap = Constraint(rule=emissions_cap_rule)

    return m

"""
python -c "
import json
from src.classical_model import build_classical_model

with open('tests/fixtures/toy_data.json') as f:
    data = json.load(f)

model = build_classical_model(data)
assert hasattr(model, 'emissions_cap'), 'La contrainte emissions_cap est absente du modèle'
print('✓ Contrainte emissions_cap présente')
print('E_max utilisé :', data['E_max'])
"
"""

