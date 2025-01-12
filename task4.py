import os
import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
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
    plt.plot(timesteps, mean_rewards_per_timestep, label="Mean reward per timestep", color="green", alpha=0.5)
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

def create_model(env, hyperparameters):
    """Crea il modello SAC con una politica MLP."""
    model = SAC(
        "MlpPolicy",
        env,
        **hyperparameters,
        verbose=1  # Verbosity level
    )
    return model

def train_model(args, env, hyperparameters):
    """Esegue l'allenamento del modello e utilizza la callback per registrare le ricompense."""

    # Crea il modello
    model = create_model(env, hyperparameters)

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

    # Train the model
    model.learn(total_timesteps=args.total_timesteps, callback=reward_logger)

    # Save the final model
    model_path = os.path.join(model_dir, args.model_name + ".zip")
    model.save(model_path)
    print(f"Modello salvato come {model_path}")

    # Salva i log
    if reward_logger.evaluations_results is not None:
        evals = reward_logger.evaluations_results
        timesteps = reward_logger.evaluations_timesteps
        np.savez(os.path.join(log_dir, "evaluations.npz"), timesteps=timesteps, rewards=evals)

    return model, reward_logger

def main():
    evaluation_file = os.path.join(log_dir, "evaluations.npz")
    # Crea l'ambiente Hopper con UDR
    env = gym.make("CustomHopper-dr-v0")

    # Define hyperparameters
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

    if args.test is None:
        model, reward_logger = train_model(args, env, hyperparameters)
        if os.path.exists(evaluation_file):
            data = np.load(evaluation_file)
            plot_rewards(data, plot_dir)
    else:
        model = SAC.load(os.path.join(model_dir, args.model_name + ".zip"), env=env)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Ricompensa media: {mean_reward}, deviazione standard: {std_reward}")

    # Test the trained policy on both source and target environments
    source_env = gym.make("CustomHopper-source-v0")
    target_env = gym.make("CustomHopper-target-v0")

    mean_reward_source, std_reward_source = evaluate_policy(model, source_env, n_eval_episodes=50)
    print(f"Source→Source: Mean reward: {mean_reward_source}, Std: {std_reward_source}")

    mean_reward_target, std_reward_target = evaluate_policy(model, target_env, n_eval_episodes=50)
    print(f"Source→Target: Mean reward: {mean_reward_target}, Std: {std_reward_target}")

    print('State space:', env.observation_space)  # spazio degli stati
    print('Action space:', env.action_space)  # spazio delle azioni
    print('Dynamics parameters:', env.get_parameters())  # parametri dinamici dell'Hopper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Modello da testare")
    parser.add_argument("--env", type=str, default="CustomHopper-dr-v0", help="Ambiente da utilizzare")
    parser.add_argument("--total_timesteps", type=int, default=5000, help="Numero totale di timesteps per l'allenamento")
    parser.add_argument("--model_name", type=str, default="sac_hopper", help="Nome del modello da salvare")
    args = parser.parse_args()
    main()