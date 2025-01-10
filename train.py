import gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from env.custom_hopper import *  # Ensure this is implemented correctly
import os

def plot_training_rewards(log_dir, plot_dir):
    """
    Legge i file di log prodotti dal training e plotta la ricompensa media.
    Salva il plot nella directory specificata.
    """
    try:
        from stable_baselines3.common.results_plotter import load_results

        results = load_results(log_dir)
        x, y = results['timesteps'], results['mean_rewards']

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="Ricompensa Media")
        plt.xlabel("Timesteps")
        plt.ylabel("Ricompensa Media")
        plt.title("Andamento Ricompensa durante il Training")
        plt.legend()
        plt.grid()

        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "training_rewards.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot salvato in: {plot_path}")
    except Exception as e:
        print(f"Errore durante il plotting: {e}")

def main():
    # Initialize the environment
    env = gym.make('CustomHopper-source-v0')

    # Print environment details
    print('State space:', env.observation_space)
    print('Action space:', env.action_space)
    print('Dynamics parameters:', env.get_parameters())

    # Directories for saving models and plots
    model_dir = "sim2real/models"
    plot_dir = "sim2real/plots"
    log_dir = "./logs_sac_hopper/"

    os.makedirs(model_dir, exist_ok=True)

    # Create the SAC model
    model = SAC(
        policy="MlpPolicy",  # Multi-Layer Perceptron policy
        env=env,  # Source environment
        verbose=1,  # Verbosity level (0: no output, 1: training info)
        tensorboard_log="./sac_hopper_tensorboard/"  # Directory for TensorBoard logs
    )

    # Define an evaluation callback to monitor performance during training
    eval_env = gym.make('CustomHopper-source-v0')  # Separate env for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,  # Evaluate every 10,000 steps
        deterministic=True,
        render=False
    )

    # Train the policy
    print("Training the SAC model...")
    model.learn(total_timesteps=100000, callback=eval_callback)  # 100k timesteps

    # Save the trained model
    model_path = os.path.join(model_dir, "sac_hopper_policy.zip")
    model.save(model_path)
    print(f"Model salvato in: {model_path}")

    # Load the trained model (optional, to demonstrate usage)
    model = SAC.load(model_path, env=env)

    # Evaluate the policy on the source environment
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, deterministic=True
    )
    print(f"Source environment: Mean reward: {mean_reward} \u00b1 {std_reward}")

    # Test the policy on the target environment
    target_env = gym.make('CustomHopper-target-v0')  # Assume the target env exists
    mean_reward, std_reward = evaluate_policy(
        model, target_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Target environment: Mean reward: {mean_reward} \u00b1 {std_reward}")

    # Plot the training rewards
    plot_training_rewards(log_dir, plot_dir)




if __name__ == '__main__':
    main()