import optuna
# import os
# import matplotlib.pyplot as plt
import os
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.callbacks import EvalCallback
import argparse

log_dir = "./sim2real/logs/"


def objective(trial):
    """Funzione obiettivo per ottimizzare gli iperparametri."""
    # Definisci gli iperparametri da ottimizzare
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    tau = trial.suggest_float("tau", 0.005, 0.05)
    ent_coef = trial.suggest_categorical("ent_coef", ["auto", 0.1, 0.01, 0.001])

    # Crea l'ambiente
    env = gym.make(args.env)

    # Crea il modello SAC
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        ent_coef=ent_coef,
        buffer_size=1000000,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10000,
        verbose=0,
    )

    # Addestra il modello (uso un numero ridotto di timesteps per velocizzare)
    model.learn(total_timesteps=30000)

    # Valuta il modello
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    # Restituisci la ricompensa media come metrica da ottimizzare
    return mean_reward


def main():
    # Configura Optuna per ottimizzare
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    # Mostra i migliori iperparametri trovati
    print("Best hyperparameters:", study.best_params)

    # Salva i risultati dello studio
    study.trials_dataframe().to_csv(os.path.join(log_dir, "optuna_results.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CustomHopper-source-v0", help="Ambiente da utilizzare")
    args = parser.parse_args()
    main()
