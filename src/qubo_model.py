# src/qubo_model.py
import dimod

def build_qubo(data, penalty_weights: dict):
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)

    valid_pairs = [
        (i, h, v) for i in data["shipments"]
        for hv in data["valid_combinations"][i]
        for h, v in [hv.split("|")]
    ]

    for i, h, v in valid_pairs:
        bqm.add_variable(f"x_{i}_{h}_{v}", data["transport_cost"][f"{i}|{h}|{v}"])

    # Affectation unique par envoi : P * (sum_x - 1)^2
    P_assign = penalty_weights["assignment"]
    for i in data["shipments"]:
        terms = [(f"x_{i}_{h}_{v}", 1) for (ii, h, v) in valid_pairs if ii == i]
        bqm.add_linear_equality_constraint(terms, lagrange_multiplier=P_assign, constant=-1)


    EMISSION_SCALE = 100
    P_emissions = penalty_weights["emissions"]
    emission_terms = [
        (f"x_{i}_{h}_{v}", round(data["emission"][f"{i}|{h}|{v}"] * EMISSION_SCALE))
        for (i, h, v) in valid_pairs
    ]
    bqm.add_linear_inequality_constraint(
        emission_terms,
        lagrange_multiplier=P_emissions,
        label="emissions_cap",
        constant=0,
        ub=round(data["E_max"] * EMISSION_SCALE),
    )

    return bqm, valid_pairs

