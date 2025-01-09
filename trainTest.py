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
    def __init__(self, log_dir, plot_dir, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.plot_dir = plot_dir
        self.episode_rewards = []
        self.episode_timesteps = []

    def _on_step(self) -> bool:
        # Verifica che "episode" sia nei locals
        if "episode" in self.locals:
            reward = self.locals["episode"]["r"]
            self.episode_rewards.append(reward)
            self.episode_timesteps.append(self.num_timesteps)
            if self.verbose > 0:
                print(f"Timestep: {self.num_timesteps}, Reward: {reward}")
        else:
            print("Chiave 'episode' non trovata nei dati locali.")
        return True

    def _on_training_end(self) -> None:
        # Salva le ricompense in un file
        rewards_file = os.path.join(self.log_dir, "rewards.csv")
        os.makedirs(self.plot_dir, exist_ok=True)

        with open(rewards_file, "w") as f:
            f.write("Timestep,Reward\n")
            for t, r in zip(self.episode_timesteps, self.episode_rewards):
                f.write(f"{t},{r}\n")

        if len(self.episode_rewards) > 0:
            self.plot_rewards()
        else:
            print("Nessuna ricompensa registrata. Impossibile creare il grafico.")

    def plot_rewards(self):
        plt.figure()
        plt.plot(self.episode_timesteps, self.episode_rewards, label="Ricompensa per episodio")
        plt.xlabel("Numero di timesteps")
        plt.ylabel("Ricompensa")
        plt.title("Ricompensa durante l'allenamento")
        plt.legend()
        plot_path = os.path.join(self.plot_dir, "rewards_plot.png")
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
    model_dir = "./sim2real/models/"
    plot_dir = "./sim2real/plots/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Crea il modello
    model = create_model(args, env)

    # Callback per registrare le ricompense
    reward_logger = RewardLoggerCallback(log_dir, plot_dir)

    # Avvia l'allenamento
    model.learn(total_timesteps=args.total_timesteps, callback=reward_logger)

    # Salva il modello
    model_path = os.path.join(model_dir, args.model_name + ".zip")
    model.save(model_path)
    print(f"Modello salvato come {model_path}")

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
