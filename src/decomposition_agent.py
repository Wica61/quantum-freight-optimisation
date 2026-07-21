# src/decomposition_agent.py
from sklearn.cluster import KMeans
import pandas as pd

def decompose_network(data: dict, n_clusters: int) -> dict:
    """Regroupe les envois par proximite geographique (coordonnees x,y du jeu
    de donnees), pour produire des sous-problemes de taille exploitable par
    un solveur QUBO. Renvoie un dict cluster_id -> sous-ensemble de 'data'."""
    ships = data["shipments"]
    df = pd.DataFrame([{"id": i, "x": s["x"], "y": s["y"]} for i, s in ships.items()])
    labels = KMeans(n_clusters=min(n_clusters, len(df)), random_state=0, n_init=10).fit_predict(df[["x", "y"]])

    clusters = {}
    for c in set(labels):
        ship_ids = df.loc[labels == c, "id"].tolist()
        sub_data = dict(data)  # copie superficielle
        sub_data["shipments"] = {i: ships[i] for i in ship_ids}
        clusters[c] = sub_data
    return clusters