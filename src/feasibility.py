# src/feasibility.py
def check_feasibility(sample: dict, data: dict) -> dict:
    """Verifie un echantillon QUBO (dict variable->0/1) contre les contraintes metier.
    Renvoie un dict avec le detail, pas juste un booleen -- utile pour deboguer
    un reglage de penalites qui echoue systematiquement sur la meme contrainte.

    NOTE sur le parsing des cles : "x_{i}_{h}_{v}" est coupe avec split("_", 3)
    (maxsplit=3), pas split("_") tout court -- indispensable des que les noms de
    vehicules contiennent eux-memes un underscore (ex. "van_1", "truck_1", utilises
    a partir de l'etape 14). Sur le jeu jouet (v1, v2, sans underscore), les deux
    methodes donnaient le meme resultat, ce qui a masque le probleme jusque-la."""
    issues = []

    # 1. chaque envoi affecte a exactement une combinaison
    for i in data["shipments"]:
        assigned = [k for k in sample if k.startswith(f"x_{i}_") and sample[k] == 1]
        if len(assigned) != 1:
            issues.append(f"{i}: {len(assigned)} affectation(s) au lieu de 1")

    # 2. activation de hub et de vehicule -- x_i,h,v=1 exige y_h=1 ET u_v=1.
    #    C'est une contrainte DURE dans le modele classique (x <= y, x <= u),
    #    mais seulement une penalite SOUCE dans le QUBO -- rien ne garantit
    #    qu'elle soit respectee sans la verifier explicitement ici.
    for k, val in sample.items():
        if val == 1 and k.startswith("x_"):
            _, i, h, v = k.split("_", 3)
            if sample.get(f"y_{h}", 0) != 1:
                issues.append(f"{k}=1 mais y_{h}={sample.get(f'y_{h}', 0)} (hub non active)")
            if sample.get(f"u_{v}", 0) != 1:
                issues.append(f"{k}=1 mais u_{v}={sample.get(f'u_{v}', 0)} (vehicule non active)")

    # 3. capacite vehicule
    load = {v: 0 for v in data["vehicles"]}
    for k, val in sample.items():
        if val == 1 and k.startswith("x_"):
            _, i, h, v = k.split("_", 3)
            load[v] += data["shipments"][i]["weight"]
    for v, cap in {v: data["vehicles"][v]["capacity"] for v in data["vehicles"]}.items():
        if load[v] > cap:
            issues.append(f"vehicule {v}: charge {load[v]} > capacite {cap}")

    # 4. plafond d'emissions
    total_emission = 0
    for k, val in sample.items():
        if val == 1 and k.startswith("x_"):
            _, i, h, v = k.split("_", 3)
            total_emission += data["emission"][f"{i}|{h}|{v}"]
    if total_emission > data["E_max"]:
        issues.append(f"emissions {total_emission:.2f} > plafond {data['E_max']}")

    return {"feasible": len(issues) == 0, "issues": issues, "total_emission": total_emission}