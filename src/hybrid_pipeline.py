# src/hybrid_pipeline.py
from src.decomposition_agent import decompose_network
from src.qubo_model import build_qubo
from src.solve_qubo import solve_qubo
from src.feasibility import check_feasibility

def recombine_solutions(cluster_solutions: list, data: dict) -> dict:
    """Fusionne les echantillons QUBO de chaque sous-probleme. Les variables x et u
    sont propres a chaque cluster (flotte partitionnee, etape 14.5) -- une simple
    union suffit. Les variables y (hubs), elles, sont PARTAGEES : si un seul cluster
    les active, elles doivent compter comme actives dans la solution globale --
    d'ou un OU logique (max) plutot qu'un simple ecrasement dict.update().

    NOTE sur le parsing : "x_{i}_{h}_{v}" est coupe avec split("_", 3) et "u_{v}"
    avec split("_", 1) -- pas split("_") tout court -- car les noms de vehicules
    a cette echelle ("van_1", "truck_1") contiennent eux-memes un underscore."""
    merged_sample = {}
    for sol in cluster_solutions:
        for k, v in sol.sample.items():
            if k.startswith("y_"):
                merged_sample[k] = max(merged_sample.get(k, 0), v)
            else:
                merged_sample[k] = v  # x et u : propres a un seul cluster, pas de conflit

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
    clusters = decompose_network(data, n_clusters)
    cluster_solutions = []
    for cluster_id, sub_data in clusters.items():
        bqm, _ = build_qubo(sub_data, penalty_weights)
        cluster_solutions.append(solve_qubo(bqm, num_reads=1000))
    return recombine_solutions(cluster_solutions, data)