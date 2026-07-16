# scripts/brute_force_reference.py
"""
Enumere les 1024 affectations possibles, filtre celles qui respectent
fenetres de temps + capacite vehicule, et donne :
  - la solution optimale sans plafond d'emissions (benchmark de cout pur)
  - la plage d'emissions parmi les solutions faisables (pour calibrer E_max)
Sert de reference independante du solveur pour valider le modele classique (etape 6)
et la reformulation QUBO (etape 9).
"""
import itertools
import json

with open("tests/fixtures/toy_data.json") as f:
    data = json.load(f)

shipments, hubs, vehicles = data["shipments"], data["hubs"], data["vehicles"]
transport_cost, emission, arrival_slot = data["transport_cost"], data["emission"], data["arrival_slot"]

options = [f"{h}|{v}" for h in hubs for v in vehicles]
feasible = []
for combo in itertools.product(options, repeat=len(shipments)):
    assign = dict(zip(shipments.keys(), combo))

    ok = all(
        shipments[i]["e"] <= arrival_slot[f"{i}|{hv.split('|')[0]}|{hv.split('|')[1]}"] <= shipments[i]["l"]
        for i, hv in assign.items()
    )
    if not ok:
        continue

    load = {v: 0 for v in vehicles}
    for i, hv in assign.items():
        load[hv.split("|")[1]] += shipments[i]["weight"]
    if any(load[v] > vehicles[v]["capacity"] for v in vehicles):
        continue

    used_hubs = {hv.split("|")[0] for hv in assign.values()}
    used_vehicles = {v for v in vehicles if load[v] > 0}
    cost = (sum(hubs[h]["cost"] for h in used_hubs)
            + sum(vehicles[v]["fixed_cost"] for v in used_vehicles)
            + sum(transport_cost[f"{i}|{hv.split('|')[0]}|{hv.split('|')[1]}"] for i, hv in assign.items()))
    total_emission = sum(emission[f"{i}|{hv.split('|')[0]}|{hv.split('|')[1]}"] for i, hv in assign.items())
    feasible.append({"cost": cost, "emission": total_emission, "assignment": assign})

feasible.sort(key=lambda s: s["cost"])
print(f"Solutions faisables (fenêtres + capacité, hors plafond émissions) : {len(feasible)} / 1024")
print(f"Meilleure solution sans plafond : coût={feasible[0]['cost']:.2f}  émissions={feasible[0]['emission']:.2f}")
print(f"Affectation : {feasible[0]['assignment']}")

with_cap = [s for s in feasible if s["emission"] <= data["E_max"]]
with_cap.sort(key=lambda s: s["cost"])
print(f"\nAvec E_max={data['E_max']} : {len(with_cap)} solutions restent faisables")
print(f"Meilleure solution sous plafond : coût={with_cap[0]['cost']:.2f}  émissions={with_cap[0]['emission']:.2f}")
print(f"Affectation : {with_cap[0]['assignment']}")

"""
python scripts/brute_force_reference.py

"""