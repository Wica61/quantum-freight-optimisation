# src/penalty_tuning.py
import optuna
from src.qubo_model import build_qubo
from src.solve_qubo import solve_qubo
from src.feasibility import check_feasibility

# Correspondance entre les noms de parametres Optuna (p_assign, ...) et les cles
# attendues par build_qubo (assignment, ...) -- utilisee a la fois dans objective()
# et dans le retour de tune_penalties(), pour ne jamais les desynchroniser.
PARAM_KEY_MAP = {
    "p_assign": "assignment",
    "p_hub": "hub_activation",
    "p_vehicle": "vehicle_activation",
    "p_capacity": "capacity",
    "p_emissions": "emissions",
}

# 🔧 Borne basse relevee de 1 a 50 : un poids de penalite a un chiffre est presque
# toujours trop faible face a des couts reels de plusieurs centaines/milliers --
# ces essais echouent quasi systematiquement et n'apprennent rien d'utile a Optuna.
def _suggest_penalty_weights(trial):
    return {full_key: trial.suggest_float(short_key, 50, 2000, log=True)
            for short_key, full_key in PARAM_KEY_MAP.items()}

def objective(trial, data):
    penalty_weights = _suggest_penalty_weights(trial)
    bqm, _ = build_qubo(data, penalty_weights)
    best = solve_qubo(bqm, num_reads=1000)
    check = check_feasibility(best.sample, data)
    return best.energy if check["feasible"] else 1e6  # penalise fortement l'infaisable

def tune_penalties(data, n_trials=100):
    study = optuna.create_study(direction="minimize")
    # 🔧 Point de depart connu-faisable (reglage manuel valide a l'etape 10,
    # ~0.4% d'ecart avec la reference) -- garantit qu'Optuna dispose d'une
    # reference faisable des le premier essai, plutot que de chercher a l'aveugle
    # dans un espace ou peu de combinaisons de 5 poids sont simultanement correctes.
    study.enqueue_trial({"p_assign": 500, "p_hub": 500, "p_vehicle": 500,
                          "p_capacity": 500, "p_emissions": 500})
    study.optimize(lambda t: objective(t, data), n_trials=n_trials)
    # study.best_params utilise les noms Optuna (p_assign, ...) -- remappage
    # obligatoire vers les cles de build_qubo avant de renvoyer, sinon KeyError
    # au premier appel de build_qubo(data, best_params) par l'appelant.
    return {PARAM_KEY_MAP[k]: v for k, v in study.best_params.items()}