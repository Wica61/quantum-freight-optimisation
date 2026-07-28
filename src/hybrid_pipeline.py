# src/hybrid_pipeline.py
from src.decomposition_agent import decompose_network
from src.qubo_model import build_qubo
from src.solve_qubo import solve_qubo
from src.classical_model import build_classical_model
from src.solve import solve_model
from src.feasibility import check_feasibility


def recombine_solutions(cluster_samples: list, data: dict) -> dict:
    """Fusionne des echantillons (dict variable->0/1) de plusieurs sous-problemes,
    qu'ils viennent d'un solveur QUBO (sampleset.first.sample) ou d'un modele
    classique resolu exactement (via _pyomo_model_to_sample ci-dessous). x et u
    sont propres a chaque cluster (flotte partitionnee, etape 14.5) -- union
    simple. y (hubs) est PARTAGE -- OU logique (max), pas un ecrasement."""
    merged_sample = {}
    for sample in cluster_samples:
        for k, v in sample.items():
            if k.startswith("y_"):
                merged_sample[k] = max(merged_sample.get(k, 0), v)
            else:
                merged_sample[k] = v

    check = check_feasibility(merged_sample, data)
    hub_cost = sum(
        data["hubs"][k.split("_", 1)[1]]["cost"]
        for k, v in merged_sample.items() if k.startswith("y_") and v == 1
    )
    vehicle_cost = sum(
        data["vehicles"][k.split("_", 1)[1]]["fixed_cost"]
        for k, v in merged_sample.items() if k.startswith("u_") and v == 1
    )
    transport_cost_total = 0
    for k, v in merged_sample.items():
        if v == 1 and k.startswith("x_"):
            _, i, h, veh = k.split("_", 3)
            transport_cost_total += data["transport_cost"][f"{i}|{h}|{veh}"]
    total_cost = hub_cost + vehicle_cost + transport_cost_total
    return {"sample": merged_sample, "cost": total_cost, "feasible": check["feasible"], "issues": check["issues"]}


def solve_hybrid(data: dict, n_clusters: int, penalty_weights: dict):
    """Pipeline hybride via QUBO/annealing -- celui qu'on debogue actuellement."""
    clusters = decompose_network(data, n_clusters)
    cluster_samples = []
    for cluster_id, sub_data in clusters.items():
        bqm, _ = build_qubo(sub_data, penalty_weights)
        best = solve_qubo(bqm, num_reads=1000)
        cluster_samples.append(best.sample)
    return recombine_solutions(cluster_samples, data)


def _pyomo_model_to_sample(model, sub_data):
    """Convertit un modele Pyomo resolu en dict {nom_variable: 0/1}, avec la
    meme convention de nommage que le QUBO (x_i_h_v, y_h, u_v) -- permet de
    reutiliser check_feasibility et recombine_solutions sans les modifier."""
    sample = {}
    for (i, h, v) in model.VALID:
        sample[f"x_{i}_{h}_{v}"] = 1 if round(model.x[i, h, v].value) == 1 else 0
    for h in sub_data["hubs"]:
        sample[f"y_{h}"] = 1 if round(model.y[h].value) == 1 else 0
    for v in sub_data["vehicles"]:
        sample[f"u_{v}"] = 1 if round(model.u[v].value) == 1 else 0
    return sample


def solve_hybrid_classical(data: dict, n_clusters: int):
    """Meme decomposition/recombinaison que solve_hybrid, mais chaque cluster
    est resolu EXACTEMENT (Gurobi) plutot que par QUBO/annealing. Sert a (1)
    obtenir un pipeline fonctionnel des maintenant, independamment du debogage
    QUBO en cours, et (2) fournir le point de comparaison "classique decompose"
    necessaire au tableau de l'etape 16."""
    clusters = decompose_network(data, n_clusters)
    cluster_samples = []
    for cluster_id, sub_data in clusters.items():
        model = build_classical_model(sub_data)
        solve_model(model, time_limit=60)
        cluster_samples.append(_pyomo_model_to_sample(model, sub_data))
    return recombine_solutions(cluster_samples, data)