import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from google.colab import drive
import zipfile
import json



# Define directories in Google Drive
model_dir = "/content/drive/MyDrive/sim2real/models/"
plot_dir = "/content/drive/MyDrive/sim2real/plots/"
log_dir = "/content/drive/MyDrive/sim2real/logs/"

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

def save_training_state(checkpoint_path, model, reward_logger, start_timesteps):
    """Save the training state including the model and logs."""
    model.save(checkpoint_path)
    state = {
        'start_timesteps': int(start_timesteps),  # Convert to standard Python int
        'evaluations_results': reward_logger.evaluations_results,  # Already a list
        'evaluations_timesteps': reward_logger.evaluations_timesteps,  # Already a list
        'evaluations_length': reward_logger.evaluations_length  # Already a list
    }
    with open(checkpoint_path + '_state.json', 'w') as f:
        json.dump(state, f)
    print(f"Checkpoint and state saved at timestep {start_timesteps}")

def load_training_state(checkpoint_path, env):
    """Load the training state including the model and logs."""
    model = SAC.load(checkpoint_path, env=env)
    with open(checkpoint_path + '_state.json', 'r') as f:
        state = json.load(f)
    start_timesteps = state['start_timesteps']
    reward_logger = EvalCallback(
        env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=30,
        deterministic=True,
        render=False
    )
    reward_logger.evaluations_results = np.array(state['evaluations_results'])  # Convert back to NumPy array
    reward_logger.evaluations_timesteps = np.array(state['evaluations_timesteps'])  # Convert back to NumPy array
    reward_logger.evaluations_length = np.array(state['evaluations_length'])  # Convert back to NumPy array
    print(f"Resuming training from timestep {start_timesteps}")
    return model, reward_logger, start_timesteps

def train_model(args, env, hyperparameters):
    """Esegue l'allenamento del modello e utilizza la callback per registrare le ricompense."""

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

    # Checkpointing
    checkpoint_interval = 1000  # Save checkpoint every 50,000 timesteps
    total_timesteps = args.total_timesteps
    start_timesteps = 0

    # Load checkpoint if available and valid
    checkpoint_path = os.path.join(model_dir, args.model_name + "_checkpoint.zip")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} exists.")
        if zipfile.is_zipfile(checkpoint_path):
            try:
                model, reward_logger, start_timesteps = load_training_state(checkpoint_path, env)
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch.")
                model = create_model(env, hyperparameters)
        else:
            print(f"Checkpoint file {checkpoint_path} is not a valid zip file.")
            model = create_model(env, hyperparameters)
    else:
        print(f"Checkpoint file {checkpoint_path} does not exist.")
        model = create_model(env, hyperparameters)

    # Train the model with checkpointing
    while start_timesteps < total_timesteps:
        remaining_timesteps = min(checkpoint_interval, total_timesteps - start_timesteps)
        model.learn(total_timesteps=remaining_timesteps, callback=reward_logger, reset_num_timesteps=False)
        start_timesteps += remaining_timesteps
        save_training_state(checkpoint_path, model, reward_logger, start_timesteps)

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
    # Crea l'ambiente Hopper
    env = gym.make(args.env)

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

    model, reward_logger = train_model(args, env, hyperparameters)
    if os.path.exists(evaluation_file):
        data = np.load(evaluation_file)
        plot_rewards(data, plot_dir)

    print('State space:', env.observation_space)  # spazio degli stati
    print('Action space:', env.action_space)  # spazio delle azioni
    print('Dynamics parameters:', env.get_parameters())  # parametri dinamici dell'Hopper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CustomHopper-source-v0", help="Ambiente da utilizzare")
    parser.add_argument("--total_timesteps", type=int, default=5000, help="Numero totale di timesteps per l'allenamento")
    parser.add_argument("--model_name", type=str, default="sac_hopper", help="Nome del modello da salvare")
    args = parser.parse_args()
    main()