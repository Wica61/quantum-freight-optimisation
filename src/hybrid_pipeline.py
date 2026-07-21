# src/hybrid_pipeline.py
from src.decomposition_agent import decompose_network
from src.qubo_model import build_qubo
from src.solve_qubo import solve_qubo
from src.feasibility import check_feasibility

def recombine_solutions(cluster_solutions: list, data: dict) -> dict:
    """Fusionne les echantillons QUBO de chaque sous-probleme en une seule
    affectation globale, et recalcule le cout total et les emissions totales
    sur l'ensemble (necessaire car le plafond d'emissions global E_max n'est
    pas garanti respecte simplement en sommant des sous-problemes resolus
    independamment -- a verifier explicitement ici)."""
    merged_sample = {}
    for sol in cluster_solutions:
        merged_sample.update(sol.sample)

    check = check_feasibility(merged_sample, data)
    total_cost = sum(
        data["transport_cost"][f"{k.split('_')[1]}|{k.split('_')[2]}|{k.split('_')[3]}"]
        for k, v in merged_sample.items() if v == 1 and k.startswith("x_")
    )
    return {"sample": merged_sample, "cost": total_cost, "feasible": check["feasible"], "issues": check["issues"]}

def solve_hybrid(data: dict, n_clusters: int, penalty_weights: dict):
    clusters = decompose_network(data, n_clusters)
    cluster_solutions = []
    for cluster_id, sub_data in clusters.items():
        bqm, _ = build_qubo(sub_data, penalty_weights)
        cluster_solutions.append(solve_qubo(bqm))
    return recombine_solutions(cluster_solutions, data)