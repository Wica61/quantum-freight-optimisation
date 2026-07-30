"""
Balayage de n_clusters avec diagnostic COMPLET avant resolution.

Trois modes de defaillance distincts, tous verifies en amont :
  1. Plafond d'emissions insuffisant : la somme des planchers decomposes
     depasse E_max. Plus on decoupe, moins chaque cluster a de choix de
     vehicules, donc plus son plancher monte -- un E_max calibre pour n
     clusters ne supporte pas n+1.
  2. Capacite insuffisante : le poids d'un cluster depasse sa capacite allouee.
  3. Envoi orphelin : tous les vehicules compatibles avec un envoi ont ete
     alloues a un autre cluster -> valid_combinations vide -> Pyomo plante
     ("constraint resolved to trivial Boolean False").

Le script reporte aussi le E_max qui SERAIT necessaire pour chaque n_clusters,
ce qui permet de decider s'il faut recalibrer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from pyomo.environ import Objective, minimize

from src.classical_model import build_classical_model
from src.solve import solve_model
from src.decomposition_agent import decompose_network
from src.hybrid_pipeline import solve_hybrid

RATIOS = {'assignment': 20.4, 'vehicle_activation': 6.3, 'emissions': 1.3,
          'hub_activation': 1.2, 'capacity': 1.0}
MARGE = 1.20

with open('tests/fixtures/scaled_data.json') as f:
    data = json.load(f)

E_MAX_REEL = data['E_max']

model = build_classical_model(data)
solve_model(model, time_limit=300)
opt = model.OBJ()

max_cost = max(max(data['transport_cost'].values()),
               max(h['cost'] for h in data['hubs'].values()),
               max(v['fixed_cost'] for v in data['vehicles'].values()))
w = {k: max_cost * r for k, r in RATIOS.items()}

print(f"\noptimum={opt:.2f}   E_max actuel={E_MAX_REEL}   max_cost={max_cost:.0f}")


def plancher_emissions(sub):
    """Plancher d'emissions reel du sous-probleme, plafond relache."""
    d = dict(sub)
    d['E_max'] = 10 ** 9
    m = build_classical_model(d)
    m.del_component(m.OBJ)

    def regle(mdl):
        return sum(d['emission'][f'{i}|{h}|{v}'] * mdl.x[i, h, v]
                   for (i, h, v) in mdl.VALID)

    m.OBJ = Objective(rule=regle, sense=minimize)
    solve_model(m, time_limit=60)
    return m.OBJ()


resume = []

for n in [2, 3, 4, 5, 6]:
    print("\n" + "=" * 70)
    print(f"n_clusters = {n}")
    print("=" * 70)

    # Decoupage obtenu avec un plafond NON contraignant, pour mesurer les
    # planchers reels sans que l'allocation ne les influence.
    data_libre = dict(data)
    data_libre['E_max'] = 10 ** 9
    clusters = decompose_network(data_libre, n_clusters=n)

    problemes = []
    planchers = {}

    for cid, sub in clusters.items():
        nship = len(sub['shipments'])
        nveh = len(sub['vehicles'])
        poids = sum(sub['shipments'][i]['weight'] for i in sub['shipments'])
        cap = sum(sub['vehicles'][v]['capacity'] for v in sub['vehicles'])
        nx = sum(len(c) for c in sub['valid_combinations'].values())

        # --- MODE 3 : envois orphelins (a verifier AVANT tout appel Pyomo) ---
        orphelins = [i for i, c in sub['valid_combinations'].items() if not c]

        # --- MODE 2 : capacite ---
        cap_ok = cap >= poids

        etat = []
        if orphelins:
            etat.append(f"ORPHELINS {orphelins[:4]}")
            problemes.append(f"cluster {cid}: {len(orphelins)} envoi(s) sans combinaison")
        if not cap_ok:
            etat.append(f"CAPACITE {poids}>{cap}")
            problemes.append(f"cluster {cid}: capacite {cap} < poids {poids}")

        print(f"  cluster {cid}: {nship:>2} envois, {nveh:>2} veh, "
              f"poids={poids:>3}/{cap:<3}, vars x={nx:>3}  "
              f"{'[' + ' | '.join(etat) + ']' if etat else '[OK]'}")

        # Le plancher n'est calculable que si le sous-probleme est bien forme
        if not orphelins and cap_ok:
            planchers[cid] = plancher_emissions(sub)

    # --- MODE 1 : plafond d'emissions ---
    if len(planchers) == len(clusters):
        somme = sum(planchers.values())
        e_requis = round(somme * MARGE, 2)
        detail = "  ".join(f"c{c}={v:.1f}" for c, v in sorted(planchers.items()))
        print(f"  planchers : {detail}")
        print(f"  somme des planchers = {somme:.2f}   "
              f"E_max actuel = {E_MAX_REEL}   "
              f"E_max requis (+{int((MARGE-1)*100)}%) = {e_requis}")
        if somme > E_MAX_REEL:
            problemes.append(
                f"somme des planchers {somme:.2f} > E_max {E_MAX_REEL} "
                f"-> decoupage impossible sous ce plafond")
    else:
        somme, e_requis = None, None
        print("  planchers non calculables (sous-probleme mal forme)")

    if problemes:
        print("  => ECARTE :")
        for p in problemes:
            print(f"       - {p}")
        resume.append({'n': n, 'verdict': 'ecarte', 'e_requis': e_requis,
                       'faisable': None, 'cause': problemes[0]})
        continue

    # --- QUBO, seulement si tout est valide ---
    taille_max = max(sum(len(c) for c in s['valid_combinations'].values())
                     for s in clusters.values())
    print(f"  taille max de sous-probleme : {taille_max} vars x "
          f"(regime favorable mesure : ~110)")

    faisables, couts, temps = 0, [], []
    for run in range(5):
        t0 = time.perf_counter()
        r = solve_hybrid(data, n_clusters=n, penalty_weights=w)
        temps.append(time.perf_counter() - t0)
        if r['feasible']:
            faisables += 1
            couts.append(r['cost'])
        print(f"    run {run+1}: faisable={r['feasible']}  "
              f"violations={len(r.get('issues', []))}  [{temps[-1]:.1f}s]")
    tm = sum(temps) / len(temps)
    if couts:
        print(f"  => {faisables}/5 faisable, meilleur {min(couts):.2f} "
              f"({min(couts)/opt-1:+.2%}), {tm:.1f}s/run")
    else:
        print(f"  => 0/5 faisable, {tm:.1f}s/run")
    resume.append({'n': n, 'verdict': 'teste', 'e_requis': e_requis,
                   'faisable': f"{faisables}/5", 'cause': ''})


# ------------------------------------------------------------------ resume
print("\n\n" + "=" * 70)
print("RESUME")
print("=" * 70)
print(f"{'n':>2}  {'verdict':<8}  {'E_max requis':>12}  {'faisable':>9}  cause")
print("-" * 70)
for r in resume:
    er = f"{r['e_requis']:.2f}" if r['e_requis'] else "n/a"
    fa = r['faisable'] or "-"
    print(f"{r['n']:>2}  {r['verdict']:<8}  {er:>12}  {fa:>9}  {r['cause'][:32]}")
print()
print(f"E_max actuel : {E_MAX_REEL}")
print("Pour tester un n_clusters ecarte pour cause d'emissions, il faut")
print("relever E_max au moins a la valeur 'E_max requis' de sa ligne.")