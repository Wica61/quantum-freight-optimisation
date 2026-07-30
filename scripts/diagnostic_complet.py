"""
DIAGNOSTIC EXHAUSTIF : pour chaque n_clusters et CHAQUE cluster, rapporte tout
ce qui peut empecher la faisabilite, dans l'ordre ou les causes s'enchainent.

Sept verifications par cluster, chacune pouvant expliquer un echec :
  1. envois orphelins (aucune combinaison valide apres filtrage vehicule)
  2. capacite allouee >= poids  (necessaire, PAS suffisant)
  3. plafond d'emissions alloue >= plancher reel du cluster
  4. faisabilite classique reelle (Gurobi) -- seul juge definitif
  5. optimum classique du cluster
  6. faisabilite QUBO DU CLUSTER SEUL  <-- jamais teste jusqu'ici
  7. faisabilite de la solution RECOMBINEE, violations par type

Le point 6 est la piece manquante : jusqu'ici on ne testait que le resultat
recombine, sans savoir si l'echec venait d'un cluster en particulier ou de la
recombinaison elle-meme.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from pyomo.environ import Objective, minimize

from src.classical_model import build_classical_model
from src.solve import solve_model
from src.qubo_model import build_qubo
from src.solve_qubo import solve_qubo
from src.feasibility import check_feasibility
from src.decomposition_agent import decompose_network
from src.hybrid_pipeline import solve_hybrid

RATIOS = {'assignment': 20.4, 'vehicle_activation': 6.3, 'emissions': 1.3,
          'hub_activation': 1.2, 'capacity': 1.0}
N_RUNS = 3
N_LIST = [2, 3, 4]

with open('tests/fixtures/scaled_data_v2.json') as f:
    data = json.load(f)


def solve_safe(model, time_limit=60):
    """Resout et renvoie (ok, valeur). ok=False si le modele n'a pas ete resolu
    -- indispensable : appeler model.OBJ() sur un modele infaisable leve une
    exception qui masque le vrai diagnostic."""
    solve_model(model, time_limit=time_limit)
    try:
        return True, model.OBJ()
    except Exception:
        return False, None


def floor_of(sub):
    d = dict(sub)
    d['E_max'] = 10 ** 9
    m = build_classical_model(d)
    m.del_component(m.OBJ)

    def rule(mdl):
        return sum(d['emission'][f'{i}|{h}|{v}'] * mdl.x[i, h, v]
                   for (i, h, v) in mdl.VALID)

    m.OBJ = Objective(rule=rule, sense=minimize)
    return solve_safe(m)


def classer(issues):
    """Repartit les violations par type -- savoir COMBIEN ne suffit pas,
    il faut savoir LESQUELLES."""
    t = {'affectation': 0, 'activation': 0, 'capacite': 0, 'emissions': 0}
    for s in issues:
        if 'affectation' in s:
            t['affectation'] += 1
        elif 'non active' in s:
            t['activation'] += 1
        elif 'charge' in s:
            t['capacite'] += 1
        elif 'emissions' in s:
            t['emissions'] += 1
    return {k: v for k, v in t.items() if v}


def poids_echelle(d):
    mc = max(max(d['transport_cost'].values()),
             max(h['cost'] for h in d['hubs'].values()),
             max(v['fixed_cost'] for v in d['vehicles'].values()))
    return {k: mc * r for k, r in RATIOS.items()}, mc


# ---------------------------------------------------------------- global
print("=" * 78)
print("DIAGNOSTIC EXHAUSTIF")
print("=" * 78)
m = build_classical_model(data)
ok, opt = solve_safe(m, 300)
print(f"Instance : {len(data['shipments'])} envois, {len(data['hubs'])} hubs, "
      f"{len(data['vehicles'])} vehicules")
print(f"E_max global = {data['E_max']}   optimum joint = {opt:.2f}")

resume = []

for n in N_LIST:
    print("\n" + "=" * 78)
    print(f"n_clusters = {n}")
    print("=" * 78)

    clusters = decompose_network(data, n_clusters=n)
    bloquants = []

    for cid, sub in clusters.items():
        print(f"\n  --- cluster {cid} " + "-" * 55)
        nship = len(sub['shipments'])
        nveh = len(sub['vehicles'])
        poids = sum(sub['shipments'][i]['weight'] for i in sub['shipments'])
        cap = sum(sub['vehicles'][v]['capacity'] for v in sub['vehicles'])
        nx = sum(len(c) for c in sub['valid_combinations'].values())
        print(f"  taille        : {nship} envois, {nveh} vehicules, {nx} variables x")

        # 1. orphelins
        orph = [i for i, c in sub['valid_combinations'].items() if not c]
        print(f"  1 orphelins   : {'AUCUN' if not orph else str(orph)}")
        if orph:
            bloquants.append(f"c{cid}: {len(orph)} envoi(s) orphelin(s)")
            continue

        # 2. capacite
        v2 = "OK" if cap >= poids else "INSUFFISANTE"
        print(f"  2 capacite    : poids {poids} / capacite {cap}  -> {v2} "
              f"(necessaire mais pas suffisant)")
        if cap < poids:
            bloquants.append(f"c{cid}: capacite {cap} < poids {poids}")
            continue

        # 3. plafond alloue vs plancher reel
        ok_f, floor = floor_of(sub)
        alloue = sub['E_max']
        if ok_f:
            marge = alloue - floor
            v3 = "OK" if marge >= 0 else "SOUS LE PLANCHER"
            print(f"  3 emissions   : alloue {alloue:.2f} / plancher reel {floor:.2f}"
                  f"  -> marge {marge:+.2f}  {v3}")
            if marge < 0:
                bloquants.append(f"c{cid}: E_max alloue {alloue:.2f} < plancher {floor:.2f}")
        else:
            print(f"  3 emissions   : plancher NON CALCULABLE (sous-probleme infaisable)")
            bloquants.append(f"c{cid}: plancher non calculable")

        # 4 + 5. faisabilite et optimum classiques
        mc = build_classical_model(sub)
        ok_c, opt_c = solve_safe(mc)
        print(f"  4 classique   : {'FAISABLE' if ok_c else 'INFAISABLE'}"
              + (f", optimum {opt_c:.2f}" if ok_c else ""))
        if not ok_c:
            bloquants.append(f"c{cid}: infaisable pour Gurobi")
            continue

        # 6. QUBO DU CLUSTER SEUL -- la verification qui manquait
        w_sub, mc_sub = poids_echelle(sub)
        faisables, viols = 0, []
        for r in range(N_RUNS):
            bqm, _ = build_qubo(sub, w_sub)
            best = solve_qubo(bqm, num_reads=1000, num_sweeps=5000)
            ch = check_feasibility(best.sample, sub)
            if ch['feasible']:
                faisables += 1
            else:
                viols.append(classer(ch['issues']))
        print(f"  6 QUBO seul   : {faisables}/{N_RUNS} faisable"
              f"   (bqm {len(bqm.variables)} vars, max_cost {mc_sub:.0f})")
        if viols:
            print(f"                  violations types : {viols}")
        if faisables == 0:
            bloquants.append(f"c{cid}: QUBO du cluster seul 0/{N_RUNS}")

    # 7. recombinaison
    print(f"\n  --- recombinaison " + "-" * 52)
    if bloquants:
        print("  Bloquants identifies AVANT recombinaison :")
        for b in bloquants:
            print(f"    - {b}")
    w, _ = poids_echelle(data)
    faisables, couts, temps, types = 0, [], [], []
    for r in range(N_RUNS):
        t0 = time.perf_counter()
        res = solve_hybrid(data, n_clusters=n, penalty_weights=w)
        temps.append(time.perf_counter() - t0)
        if res['feasible']:
            faisables += 1
            couts.append(res['cost'])
        else:
            types.append(classer(res['issues']))
    tm = sum(temps) / len(temps)
    print(f"  7 recombine   : {faisables}/{N_RUNS} faisable, {tm:.1f}s/run")
    if types:
        print(f"                  violations types : {types}")
    if couts:
        print(f"                  meilleur cout {min(couts):.2f} "
              f"({min(couts)/opt-1:+.2%})")

    resume.append({'n': n, 'faisable': f"{faisables}/{N_RUNS}",
                   'bloquants': len(bloquants),
                   'detail': bloquants[0] if bloquants else '-'})

print("\n\n" + "=" * 78)
print("RESUME")
print("=" * 78)
print(f"{'n':>2}  {'recombine':>10}  {'bloquants':>9}  premier bloquant")
print("-" * 78)
for r in resume:
    print(f"{r['n']:>2}  {r['faisable']:>10}  {r['bloquants']:>9}  {r['detail'][:44]}")