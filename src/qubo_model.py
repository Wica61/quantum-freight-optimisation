# src/qubo_model.py
import dimod

EMISSION_SCALE = 10

def build_qubo(data, penalty_weights: dict):
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)

    valid_pairs = [
        (i, h, v) for i in data["shipments"]
        for hv in data["valid_combinations"][i]
        for h, v in [hv.split("|")]
    ]

    for i, h, v in valid_pairs:
        bqm.add_variable(f"x_{i}_{h}_{v}", data["transport_cost"][f"{i}|{h}|{v}"])
    for h in data["hubs"]:
        bqm.add_variable(f"y_{h}", data["hubs"][h]["cost"])
    for v in data["vehicles"]:
        bqm.add_variable(f"u_{v}", data["vehicles"][v]["fixed_cost"])

    P_assign = penalty_weights["assignment"]
    for i in data["shipments"]:
        terms = [(f"x_{i}_{h}_{v}", 1) for (ii, h, v) in valid_pairs if ii == i]
        bqm.add_linear_equality_constraint(terms, lagrange_multiplier=P_assign, constant=-1)

    P_hub = penalty_weights["hub_activation"]
    for (i, h, v) in valid_pairs:
        bqm.add_linear(f"x_{i}_{h}_{v}", P_hub)
        bqm.add_quadratic(f"x_{i}_{h}_{v}", f"y_{h}", -P_hub)

    P_veh = penalty_weights["vehicle_activation"]
    for (i, h, v) in valid_pairs:
        bqm.add_linear(f"x_{i}_{h}_{v}", P_veh)
        bqm.add_quadratic(f"x_{i}_{h}_{v}", f"u_{v}", -P_veh)

    P_capacity = penalty_weights["capacity"]
    for v in data["vehicles"]:
        terms = [(f"x_{i}_{h}_{v}", data["shipments"][i]["weight"])
                 for (i, h, vv) in valid_pairs if vv == v]
        terms.append((f"u_{v}", -data["vehicles"][v]["capacity"]))
        bqm.add_linear_inequality_constraint(
            terms, lagrange_multiplier=P_capacity, label=f"capacity_{v}", constant=0, ub=0
        )

    # EMISSION_SCALE=100 avait ete introduit pour supprimer un avertissement de
    # dimod sur les coefficients fractionnaires. Effet de bord grave : la
    # penalite d'une inegalite est QUADRATIQUE dans le residu, donc mettre les
    # coefficients a l'echelle x100 multiplie la penalite par 100^2 = 10 000.
    # Resultat : abandonner un envoi (cout P) devenait 2500x moins cher que
    # d'ajuster les emissions -- le solveur sacrifiait des envois.
    # On arrondit donc a l'unite : la perte de precision (< 1 unite d'emissions
    # sur un plafond de ~200) est negligeable, et le paysage d'energie redevient
    # exploitable (8 bits d'ecart au lieu de 15).
    # EMISSION_SCALE = 1

    # EMISSION_SCALE=1 introduisait une erreur d'arrondi de +/-0.49 par terme, soit
    # jusqu'a +7 cumules sur 30 envois actifs. Negligeable a 25 % de marge, mais
    # decisif des qu'on resserre E_max (balayage a venir : marge ~6 unites a 1.02x).
    # On monte a 10 -> erreur cumulee < 1.5.
    # ATTENTION : la penalite d'une inegalite est QUADRATIQUE dans le residu.
    # Multiplier les coefficients par 10 multiplie la penalite par 100 ; il faut
    # donc diviser le poids d'emissions d'autant pour conserver l'equilibre entre
    # les cinq familles de penalites (c'est ce qui avait fait echouer SCALE=100).
    
    P_emissions = penalty_weights["emissions"]
    emission_terms = [
        (f"x_{i}_{h}_{v}", round(data["emission"][f"{i}|{h}|{v}"] * EMISSION_SCALE))
        for (i, h, v) in valid_pairs
    ]
    bqm.add_linear_inequality_constraint(
        emission_terms,
        lagrange_multiplier=P_emissions / (EMISSION_SCALE ** 2),
        label="emissions_cap",
        constant=0,
        ub=int(data["E_max"] * EMISSION_SCALE),   # int() = arrondi vers le bas, conservateur
    )
    return bqm, valid_pairs