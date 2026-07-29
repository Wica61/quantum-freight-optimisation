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
    """Convertit un modele Pyomo RESOLU en dict {nom_variable: 0/1}.

    Renvoie None si le modele n'a pas ete resolu (sous-probleme infaisable) :
    dans ce cas Pyomo laisse toutes les valeurs a None, et tenter de les lire
    plante. L'appelant DOIT verifier ce retour."""
    sample = {}
    for (i, h, v) in model.VALID:
        val = model.x[i, h, v].value
        if val is None:
            return None          # sous-probleme non resolu
        sample[f"x_{i}_{h}_{v}"] = 1 if round(val) == 1 else 0
    for h in sub_data["hubs"]:
        val = model.y[h].value
        if val is None:
            return None
        sample[f"y_{h}"] = 1 if round(val) == 1 else 0
    for v in sub_data["vehicles"]:
        val = model.u[v].value
        if val is None:
            return None
        sample[f"u_{v}"] = 1 if round(val) == 1 else 0
    return sample


def solve_hybrid_classical(data: dict, n_clusters: int):
    """Meme decomposition/recombinaison que solve_hybrid, mais chaque cluster
    est resolu EXACTEMENT (Gurobi) plutot que par QUBO/annealing.

    Si UN SEUL cluster est infaisable pris isolement, la decomposition entiere
    est invalide : on le signale proprement au lieu de planter. Cela arrive
    quand le decoupage est trop agressif pour la taille du probleme (ex. 2
    clusters sur un jeu a 5 envois et 2 vehicules => 1 vehicule par cluster)."""
    clusters = decompose_network(data, n_clusters)
    cluster_samples = []
    for cluster_id, sub_data in clusters.items():
        model = build_classical_model(sub_data)
        solve_model(model, time_limit=60)
        sample = _pyomo_model_to_sample(model, sub_data)
        if sample is None:
            return {
                "sample": {},
                "cost": None,
                "feasible": False,
                "issues": [
                    f"cluster {cluster_id} ({len(sub_data['shipments'])} shipments, "
                    f"{len(sub_data['vehicles'])} vehicles) is infeasible on its own "
                    f"-- decomposition into {n_clusters} clusters is not viable here"
                ],
                "failed_cluster": cluster_id,
            }
        cluster_samples.append(sample)
    return recombine_solutions(cluster_samples, data)