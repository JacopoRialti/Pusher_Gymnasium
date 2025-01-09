import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggerCallback(BaseCallback):
    """
    Callback per registrare le ricompense durante l'allenamento.
    Salva i risultati in un file CSV e crea un grafico delle ricompense alla fine.
    """
    def __init__(self, log_dir, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_timesteps = []

    def _on_step(self) -> bool:
        # Aggiungi la ricompensa dell'episodio corrente
        if "episode" in self.locals:
            self.episode_rewards.append(self.locals["episode"]["r"])
            self.episode_timesteps.append(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        # Salva le ricompense in un file
        rewards_file = os.path.join(self.log_dir, "rewards.csv")
        with open(rewards_file, "w") as f:
            f.write("Timestep,Reward\n")
            for t, r in zip(self.episode_timesteps, self.episode_rewards):
                f.write(f"{t},{r}\n")

        # Crea il grafico delle ricompense
        self.plot_rewards()

    def plot_rewards(self):
        plt.figure()
        plt.plot(self.episode_timesteps, self.episode_rewards, label="Ricompensa per episodio")
        plt.xlabel("Numero di timesteps")
        plt.ylabel("Ricompensa")
        plt.title("Ricompensa durante l'allenamento")
        plt.legend()
        plot_path = os.path.join(self.log_dir, "rewards_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Grafico delle ricompense salvato in {plot_path}")


def create_model(args, env):
    """Crea il modello SAC con una politica MLP."""
    model = SAC("MlpPolicy", env, verbose=1)
    return model


def train_model(args, env):
    """Esegue l'allenamento del modello e utilizza la callback per registrare le ricompense."""
    log_dir = "./tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    # Crea il modello
    model = create_model(args, env)

    # Callback per registrare le ricompense
    reward_logger = RewardLoggerCallback(log_dir)

    # Avvia l'allenamento
    model.learn(total_timesteps=args.total_timesteps, callback=reward_logger)

    # Salva il modello
    model.save(args.model_name)
    print(f"Modello salvato come {args.model_name}")

    return model


def main():
    # Crea l'ambiente Hopper
    env = gym.make(args.env)

    if args.test is None:
        model = train_model(args, env)
    else:
        model = load_model(args, env)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Ricompensa media: {mean_reward}, deviazione standard: {std_reward}")

    print('State space:', env.observation_space)  # spazio degli stati
    print('Action space:', env.action_space)  # spazio delle azioni
    print('Dynamics parameters:', env.get_parameters())  # parametri dinamici dell'Hopper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Modello da testare")
    parser.add_argument("--env", type=str, default="CustomHopper-source-v0", help="Ambiente da utilizzare")
    parser.add_argument("--total_timesteps", type=int, default=5000, help="Numero totale di timesteps per l'allenamento")
    parser.add_argument("--model_name", type=str, default="sac_hopper", help="Nome del modello da salvare")
    args = parser.parse_args()
    main()
