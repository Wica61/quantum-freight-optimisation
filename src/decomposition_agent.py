# src/decomposition_agent.py (version étendue)
from sklearn.cluster import KMeans
import pandas as pd

def decompose_network(data: dict, n_clusters: int) -> dict:
    """Partitionne les envois (proximite geographique), la flotte de vehicules
    (proportionnellement au poids de chaque cluster) ET le plafond d'emissions
    (proportionnellement au nombre d'envois) -- aucune ressource partagee n'est
    jamais dupliquee entre clusters, ce qui elimine par construction les deux
    bugs de ressource partagee trouves aux etapes 13 et 14.6."""
    ships = data["shipments"]
    df = pd.DataFrame([{"id": i, "x": s["x"], "y": s["y"], "weight": s["weight"]} for i, s in ships.items()])
    labels = KMeans(n_clusters=min(n_clusters, len(df)), random_state=0, n_init=10).fit_predict(df[["x", "y"]])
    df["cluster"] = labels

    cluster_weight = df.groupby("cluster")["weight"].sum().to_dict()
    cluster_ids = sorted(cluster_weight.keys())

    # Partition gloutonne de la flotte : chaque vehicule va au cluster le plus
    # "sous-dote" relativement a son poids, en traitant les plus gros vehicules
    # en premier (evite de laisser les petits clusters avec tous les petits vehicules).
    vehicles_sorted = sorted(data["vehicles"].items(), key=lambda kv: -kv[1]["capacity"])
    allocated_vehicles = {c: [] for c in cluster_ids}
    allocated_capacity = {c: 0 for c in cluster_ids}
    for vname, vdata in vehicles_sorted:
        target = min(cluster_ids, key=lambda c: allocated_capacity[c] / max(cluster_weight[c], 1))
        allocated_vehicles[target].append(vname)
        allocated_capacity[target] += vdata["capacity"]

    # Partition proportionnelle du plafond d'emissions -- proportionnelle au nombre
    # d'envois (l'emission d'un envoi depend de la distance, pas de son poids,
    # contrairement a la capacite vehicule qui elle depend bien du poids).
    total_shipments = len(ships)
    cluster_n_shipments = df.groupby("cluster").size().to_dict()

    clusters = {}
    for c in cluster_ids:
        ship_ids = df.loc[df["cluster"] == c, "id"].tolist()
        sub_data = dict(data)
        sub_data["shipments"] = {i: ships[i] for i in ship_ids}
        sub_data["vehicles"] = {v: data["vehicles"][v] for v in allocated_vehicles[c]}
        # Les combinaisons valides doivent aussi etre restreintes aux vehicules
        # alloues a CE cluster -- sinon x pourrait encore referencer un vehicule
        # qui appartient a un autre cluster.
        sub_data["valid_combinations"] = {
            i: [hv for hv in data["valid_combinations"][i] if hv.split("|")[1] in allocated_vehicles[c]]
            for i in ship_ids
        }
        # Plafond d'emissions local, proportionnel au nombre d'envois de ce cluster --
        # empeche chaque sous-QUBO de "voir" le plafond global en entier.
        sub_data["E_max"] = round(data["E_max"] * cluster_n_shipments[c] / total_shipments, 2)
        clusters[c] = sub_data

    return clusters