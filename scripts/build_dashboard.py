#!/usr/bin/env python
"""
Assemble results/dashboard.html, autonome, a partir de deux sources.

    scripts/export_dashboard.py   (le votre, inchange)
        fixture  ->  results/dashboard.json      partie DETERMINISTE
                                                 instance, partitions KMeans,
                                                 taille reelle des sous-QUBO

    scripts/build_dashboard.py    (ce fichier)
        dashboard.json + measures.json  ->  results/dashboard.html

Ce script n'ecrase PAS votre export_dashboard.py : il le prolonge. Les deux se
lancent a la suite. Le HTML produit embarque ses donnees dans une balise
<script>, donc il s'ouvre par double-clic -- plus de fetch, plus de CORS.

Usage :
    python scripts/export_dashboard.py && python scripts/build_dashboard.py
    open results/dashboard.html
"""
import argparse
import json
import re
import sys
from pathlib import Path

DASHBOARD_JSON = Path("results/dashboard.json")
MEASURES = Path("results/measures.json")
TEMPLATE = Path("results/dashboard.html")
OUT = Path("results/dashboard.html")

BLOC = re.compile(
    r'(<script id="measures" type="application/json">)(.*?)(</script>)', re.S
)

# Sections reprises telles quelles depuis measures.json.
DIRECTES = [
    "baseline_jointe", "balayage_emax", "bimodalite", "plafond_n",
    "sweeps", "loi_cout", "hubs_mecanisme", "allocation_carbone",
    "emission_scale", "temps", "ouvert",
]

# Un meme resultat a ete enregistre sous plusieurs noms au fil de la campagne.
ALIAS = {"campagne": ["campagne", "campagne_runs", "runs_par_n"],
         "plafond_n": ["plafond_n", "fenetre_n"]}

# Champ identifiant une ligne : sert a fusionner ligne a ligne plutot qu'a
# ecraser. Une mesure partielle ne doit jamais faire disparaitre une colonne.
CLE_LIGNE = {"campagne": ("n",), "plafond_n": ("n",), "balayage_emax": ("mult",),
             "sweeps": ("n", "sweeps"), "hubs_mecanisme": ("mult",)}

SYNONYMES = {"faisable": "faisables"}


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------
def _id_ligne(r, champs):
    return tuple(r.get(c) for c in champs)


def _fusionne_lignes(base, neuf, champs):
    par_id = {_id_ligne(r, champs): dict(r) for r in base if isinstance(r, dict)}
    for r in neuf:
        if not isinstance(r, dict) or None in _id_ligne(r, champs):
            continue
        cible = par_id.setdefault(_id_ligne(r, champs), {})
        for k, v in r.items():
            if v is not None:
                cible[SYNONYMES.get(k, k)] = v
    return [par_id[k] for k in sorted(par_id, key=lambda t: tuple(x or 0 for x in t))]


def _normalise_plafond(v):
    if isinstance(v, list):
        return v
    planchers = (v or {}).get("planchers") or {}
    return [{"n": int(n), "planchers": p}
            for n, p in sorted(planchers.items(), key=lambda kv: int(kv[0]))] or None


def _normalise_bimodalite(v, base):
    v = dict(v)
    if "modes" in v or "par_vehicules" not in v:
        return v
    couts = {str(m["veh"]): m.get("cout") for m in (base or {}).get("modes", [])}
    v["modes"] = [{"veh": int(k), "runs": n, "cout": couts.get(k, "—")}
                  for k, n in sorted(v.pop("par_vehicules").items(), key=lambda kv: int(kv[0]))]
    for k in ("ecart_modes", "transport"):
        if k not in v and base and k in base:
            v[k] = base[k]
    return v


# --------------------------------------------------------------------------
# Partie deterministe : dashboard.json -> section "carte"
# --------------------------------------------------------------------------
def extrait_carte(doc):
    """Reformate la sortie de export_dashboard.py pour la plaque « instance ».

    Attention aux deux comptages de variables, qui ne mesurent pas la meme chose :
      n_variables_total  = binaires du modele classique (x + y + u)
      variables_jointes  = variables du BQM joint, ecarts compris
    Le second ne doit jamais etre ecrase par le premier.
    """
    inst = doc.get("instance") or {}
    parts = doc.get("partitions") or {}
    if not inst.get("envois") or not parts:
        return None, {}

    carte = {
        "hubs": [{"id": h, "x": d["x"], "y": d["y"], "cost": d.get("cost")}
                 for h, d in (inst.get("hubs") or {}).items()],
        "envois": [{"id": e["id"], "x": e["x"], "y": e["y"], "w": e["w"]}
                   for e in inst["envois"]],
        "partitions": {n: {"labels": p["labels"], "clusters": p["clusters"],
                           "bqm_max": p["bqm_max"]}
                       for n, p in parts.items()},
    }
    maj = {k: inst[k] for k in ("E_max",) if k in inst}
    if "n_envois" in inst:
        maj["envois"] = inst["n_envois"]
    if "n_vehicules" in inst:
        maj["vehicules"] = inst["n_vehicules"]
    if "n_variables_total" in inst:
        maj["variables_classiques"] = inst["n_variables_total"]
    for k in ("marge_capacite", "poids_total", "capacite_totale"):
        if k in inst:
            maj[k] = inst[k]
    return carte, maj


def construire(mes, doc, embarque):
    out = dict(embarque)
    journal = []

    for cle in DIRECTES + ["campagne"]:
        noms = ALIAS.get(cle, [cle])
        trouve = next((n for n in noms if n in mes), None)
        sources = [mes[trouve]] if trouve else []
        if cle == "campagne" and isinstance(mes.get("surcout_n5_n6"), list):
            sources.append(mes["surcout_n5_n6"])
            trouve = trouve or "surcout_n5_n6"
        if not sources:
            journal.append((".", cle, "conserve l'instantane embarque"))
            continue

        valeur = sources[0]
        if cle == "plafond_n":
            valeur = _normalise_plafond(valeur) or out.get(cle)
        if cle == "bimodalite":
            valeur = _normalise_bimodalite(valeur, out.get(cle))

        if cle in CLE_LIGNE and isinstance(out.get(cle), list):
            for src in sources:
                if isinstance(src, list):
                    out[cle] = _fusionne_lignes(out[cle], src, CLE_LIGNE[cle])
            detail = f"{trouve} — {len(out[cle])} lignes"
        else:
            out[cle] = valeur
            detail = trouve
        journal.append(("+", cle, detail))

    carte, maj = extrait_carte(doc)
    if carte:
        out["carte"] = carte
        out["instance"] = {**out.get("instance", {}), **maj}
        journal.append(("+", "carte", f"{len(carte['envois'])} envois, "
                                      f"{len(carte['hubs'])} hubs, "
                                      f"n = {', '.join(sorted(carte['partitions']))}"))
    else:
        journal.append((".", "carte", "dashboard.json absent ou incomplet"))

    connues = set(DIRECTES) | set(sum(ALIAS.values(), [])) | {"surcout_n5_n6", "surcout", "meta"}
    for cle in sorted(set(mes) - connues):
        journal.append(("?", cle, "aucune plaque ne l'affiche"))

    if "meta" in mes:
        out["meta"] = {**out.get("meta", {}), **mes["meta"]}
    return out, journal


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dashboard-json", type=Path, default=DASHBOARD_JSON)
    ap.add_argument("--measures", type=Path, default=MEASURES)
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    if not a.template.exists():
        sys.exit(f"gabarit introuvable : {a.template}\n"
                 f"Copiez dashboard.html dans results/ avant de lancer ce script.")

    html = a.template.read_text(encoding="utf-8")
    bloc = BLOC.search(html)
    if not bloc:
        sys.exit(f'balise <script id="measures"> absente de {a.template} — '
                 f'ce n\'est pas le bon gabarit.')
    embarque = json.loads(bloc.group(2))

    mes = json.loads(a.measures.read_text()) if a.measures.exists() else {}
    doc = json.loads(a.dashboard_json.read_text()) if a.dashboard_json.exists() else {}
    if not a.measures.exists():
        print(f"! {a.measures} absent")
    if not a.dashboard_json.exists():
        print(f"! {a.dashboard_json} absent — lancez d'abord export_dashboard.py")

    donnees, journal = construire(mes, doc, embarque)

    largeur = max((len(c) for _, c, _ in journal), default=10)
    for marque, cle, detail in journal:
        print(f"  {marque} {cle:<{largeur}}  {detail}")

    charge = json.dumps(donnees, indent=2, ensure_ascii=False)
    html = html[:bloc.start(2)] + "\n" + charge + "\n" + html[bloc.end(2):]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(html, encoding="utf-8")

    pris = sum(1 for m, _, _ in journal if m == "+")
    print(f"\n-> {a.out}  ({len(html) // 1024} Ko, {pris} sections relues)")
    print("   Ouvrez-le par double-clic : aucun serveur local n'est necessaire.")


if __name__ == "__main__":
    main()