import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from pyvirtualdisplay import Display
import imageio

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


    # Adjust timesteps to match the length of mean_rewards
    adjusted_timesteps = timesteps[window_size-1:]
   
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards_per_timestep, label="Mean reward per timestep", alpha=0.5)
    plt.plot(adjusted_timesteps, mean_rewards, label="Mean reward (50-timestep MA)", color='blue')
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

    # Print the array of mean rewards per timestep
    print("Mean rewards per timestep:", mean_rewards_per_timestep)


def main():
    plot_dir = "project-sim2real-rialti-giunti-gjinaj/sim2real/plots"
    model_dir = "project-sim2real-rialti-giunti-gjinaj/sim2real/models"
    log_dir = "logs"

    # Determine the environment based on the argument
    if args.env == "source":
        env_name = "CustomHopper-source-v0"
    elif args.env == "target":
        env_name = "CustomHopper-target-v0"
    else:
        raise ValueError("Invalid environment specified. Use 'source' or 'target'.")

    env = gym.make(env_name)
    model_path = os.path.join(model_dir, args.model_name)
    model= SAC.load(model_path)

    # Test the model on the specified environment
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print(f"Mean reward: {mean_reward}, Std: {std_reward}")

    if args.render:
        render_video(model, env)

    print('State space:', env.observation_space)  # spazio degli stati
    print('Action space:', env.action_space)  # spazio delle azioni
    print('Dynamics parameters:', env.get_parameters())  # parametri dinamici dell'Hopper



def render_video(model, env):
    display = Display(visible=0, size=(1400, 900))
    display.start()
    n_episodes = 5  # Set to 1 for video recording
    frames = []
    
    for ep in range(n_episodes):  
        done = False
        state = env.reset()  # Reset environment to initial state

        while not done:  # Until the episode is over
            action = model.predict(state, deterministic=True)[0]  # Use the model to predict the action
            state, reward, done, info = env.step(action)  # Step the simulator to the next timestep

            
            frame = env.render(mode='rgb_array')
            frames.append(frame)

    video_path = os.path.join("sim2realtmp/plots", f"{args.model_name}_on_{args.env}.mp4")
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Video saved at {video_path}")

    display.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=['source', 'target'], default=None, help="Ambiente da utilizzare (source o target)")
    parser.add_argument("--total_timesteps", type=int, default=5000, help="Numero totale di timesteps per l'allenamento")
    parser.add_argument("--model_name", type=str, default="sac_hopper", help="Nome del modello da salvare")
    parser.add_argument("--render", type=bool, default=False, help="Se True, renderizza l'ambiente")
    args = parser.parse_args()
    main()