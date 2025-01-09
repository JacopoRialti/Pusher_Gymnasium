import gym
import argparse
import matplotlib.pyplot as plt
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import os


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()

def main(model_save_name, plot_save_name, total_timesteps):
    # Create the Hopper environment
    env = gym.make('CustomHopper-source-v0')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

    # Initialize the SAC model
    model = SAC('MlpPolicy', env, verbose=1)

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save(model_save_name)

    # Plot the results
    plot_results("sim2real/plots/", title="Training Progress")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Test the policy on the target environment
    target_env = gym.make('CustomHopper-target-v0')
    mean_reward, std_reward = evaluate_policy(model, target_env, n_eval_episodes=10)
    print(f"Mean reward on target environment: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a SAC model on the CustomHopper environment.')
    parser.add_argument('--model_name', type=str, default='model_name', help='The name to save the trained model.')
    parser.add_argument('--train_plot', type=str, default='train_plot.png', help='The name to save the training plot.')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='The total number of timesteps to train on.')
    args = parser.parse_args()
    main(args.model_name, args.train_plot, args.total_timesteps)