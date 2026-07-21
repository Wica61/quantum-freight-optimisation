# src/feasibility.py
def check_feasibility(sample: dict, data: dict) -> dict:
    """Verifie un echantillon QUBO (dict variable->0/1) contre les contraintes metier.
    Renvoie un dict avec le detail, pas juste un booleen -- utile pour deboguer
    un reglage de penalites qui echoue systematiquement sur la meme contrainte."""
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
            _, i, h, v = k.split("_")
            if sample.get(f"y_{h}", 0) != 1:
                issues.append(f"{k}=1 mais y_{h}={sample.get(f'y_{h}', 0)} (hub non active)")
            if sample.get(f"u_{v}", 0) != 1:
                issues.append(f"{k}=1 mais u_{v}={sample.get(f'u_{v}', 0)} (vehicule non active)")

    # 3. capacite vehicule
    load = {v: 0 for v in data["vehicles"]}
    for k, val in sample.items():
        if val == 1 and k.startswith("x_"):
            _, i, h, v = k.split("_")
            load[v] += data["shipments"][i]["weight"]
    for v, cap in {v: data["vehicles"][v]["capacity"] for v in data["vehicles"]}.items():
        if load[v] > cap:
            issues.append(f"vehicule {v}: charge {load[v]} > capacite {cap}")

    # 4. plafond d'emissions
    total_emission = sum(
        data["emission"][f"{k.split('_')[1]}|{k.split('_')[2]}|{k.split('_')[3]}"]
        for k, val in sample.items() if val == 1 and k.startswith("x_")
    )
    if total_emission > data["E_max"]:
        issues.append(f"emissions {total_emission:.2f} > plafond {data['E_max']}")

    return {"feasible": len(issues) == 0, "issues": issues, "total_emission": total_emission}