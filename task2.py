import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


model_dir = "./sim2real/models/"
plot_dir = "./sim2real/plots/"
log_dir = "./sim2real/logs/"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)



def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_rewards(data, plot_dir):
    """
    Generates a plot of episodic rewards and saves it to the specified directory.
    """
    timesteps = data['timesteps']
    rewards = data['rewards']

    # Calculate the mean reward for each timestep across the 5 episodes
    mean_rewards_per_timestep = rewards.mean(axis=1)

    # Calculate the moving average of the mean rewards with a window size of 50
    window_size = 50
    mean_rewards = moving_average(mean_rewards_per_timestep, window_size)
   
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards_per_timestep, label="Mean reward per timestep",color = "green", alpha=0.5)
    plt.plot(timesteps[window_size-1:], mean_rewards, label="Mean reward (50-timestep MA)", color='blue')
    plt.xlabel("Number of timesteps")
    plt.ylabel("Reward")
    plt.title("Reward during training")
    plt.legend()
    plt.grid()

    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "rewards_plot.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Reward plot saved in {plot_path}")


def create_model(args, env):
    """Crea il modello SAC con una politica MLP."""
    #Stiven Crea modelli, con hyperparameters diversi
    hyperparameters = {
        "learning_rate": 3e-4,
        "batch_size": 256,
        "buffer_size": 1000000,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_starts": 10000,
        "ent_coef": 'auto'
    }

    model = SAC(
        "MlpPolicy",
        env,
        **hyperparameters,
        verbose=1  # Verbosity level
        )
    return model


def train_model(args, env):
    """Esegue l'allenamento del modello e utilizza la callback per registrare le ricompense."""

    # Crea il modello
    model = create_model(args, env)

    # Crea ambiente per la valutazione
    eval_env = gym.make(args.env)

    reward_logger = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=30,
        deterministic=True,
        render=False
    )

    
    # Avvia l'allenamento
    model.learn(total_timesteps=args.total_timesteps, callback=reward_logger)

    # Salva il modello
    model_path = os.path.join(model_dir, args.model_name + ".zip")
    model.save(model_path)
    print(f"Modello salvato come {model_path}")

    #Salva i log
    if reward_logger.evaluations_results is not None:
        evals = reward_logger.evaluations_results
        timesteps = reward_logger.evaluations_timesteps
        np.savez(os.path.join(log_dir, "evaluations.npz"), timesteps=timesteps, rewards=evals)

    return model, reward_logger


def main():
    evaluation_file = os.path.join(log_dir, "evaluations.npz")
    # Crea l'ambiente Hopper
    env = gym.make(args.env)


    if args.test is None:
        model,reward_logger = train_model(args, env)
        if os.path.exists(evaluation_file):
            data = np.load(evaluation_file)
            # timesteps, rewards = data["timesteps"], data["rewards"]
            plot_rewards(data, plot_dir)
    else:
        model = SAC.load_model(args, env)
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
