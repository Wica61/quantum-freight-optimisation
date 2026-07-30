# src/decomposition_agent.py  -- VERSION FINALE
from sklearn.cluster import KMeans
import pandas as pd
from pyomo.environ import Objective, minimize
from src.classical_model import build_classical_model
from src.solve import solve_model


def _emissions_floor(sub_data):
    """Plancher d'emissions REEL de ce sous-probleme, avec SES vehicules alloues.
    Indispensable : partitionner E_max au prorata du nombre d'envois suppose que
    tous les envois emettent autant, ce qui est faux -- un cluster geographiquement
    eloigne des hubs peut recevoir moins que son minimum atteignable, ce qui le
    rend infaisable par construction."""
    d = dict(sub_data)
    d["E_max"] = 10 ** 9
    model = build_classical_model(d)
    model.del_component(model.OBJ)

    def rule(mdl):
        return sum(d["emission"][f"{i}|{h}|{v}"] * mdl.x[i, h, v] for (i, h, v) in mdl.VALID)

    model.OBJ = Objective(rule=rule, sense=minimize)
    solve_model(model, time_limit=60)
    try:
        return model.OBJ()
    except Exception:
        return None          # sous-probleme mal forme


def decompose_network(data: dict, n_clusters: int) -> dict:
    """Partitionne les envois (KMeans geographique), la flotte (glouton
    proportionnel au poids) et le plafond d'emissions (proportionnel au
    PLANCHER REEL de chaque cluster)."""
    ships = data["shipments"]
    df = pd.DataFrame([{"id": i, "x": s["x"], "y": s["y"], "weight": s["weight"]}
                       for i, s in ships.items()])
    labels = KMeans(n_clusters=min(n_clusters, len(df)), random_state=0,
                    n_init=10).fit_predict(df[["x", "y"]])
    df["cluster"] = labels

    cluster_weight = df.groupby("cluster")["weight"].sum().to_dict()
    cluster_ids = sorted(cluster_weight.keys())

    vehicles_sorted = sorted(data["vehicles"].items(), key=lambda kv: -kv[1]["capacity"])
    allocated = {c: [] for c in cluster_ids}
    alloc_cap = {c: 0 for c in cluster_ids}
    for vname, vdata in vehicles_sorted:
        target = min(cluster_ids, key=lambda c: alloc_cap[c] / max(cluster_weight[c], 1))
        allocated[target].append(vname)
        alloc_cap[target] += vdata["capacity"]

    prelim = {}
    for c in cluster_ids:
        ship_ids = df.loc[df["cluster"] == c, "id"].tolist()
        sub = dict(data)
        sub["shipments"] = {i: ships[i] for i in ship_ids}
        sub["vehicles"] = {v: data["vehicles"][v] for v in allocated[c]}
        sub["valid_combinations"] = {
            i: [hv for hv in data["valid_combinations"][i]
                if hv.split("|")[1] in allocated[c]]
            for i in ship_ids
        }
        prelim[c] = sub

    # Planchers reels, puis repartition PROPORTIONNELLE A CES PLANCHERS
    floors = {c: _emissions_floor(sub) for c, sub in prelim.items()}
    valides = {c: f for c, f in floors.items() if f is not None}
    total = sum(valides.values()) if valides else 0

    clusters = {}
    for c in cluster_ids:
        sub = prelim[c]
        if floors[c] is not None and total > 0:
            sub["E_max"] = round(data["E_max"] * floors[c] / total, 2)
        else:
            sub["E_max"] = data["E_max"]      # sous-probleme mal forme : on ne restreint pas
        sub["_floor"] = floors[c]             # conserve pour le diagnostic
        clusters[c] = sub

    return clusters