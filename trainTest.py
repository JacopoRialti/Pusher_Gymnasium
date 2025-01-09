import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.callbacks import BaseCallback  # Callback class for logging rewards
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy



def create_model(args, env):
    model = SAC("MlpPolicy", env, verbose=1)
    return model

def load_model(args, env):
    model = SAC.load(args.test, env=env)
    return model

def train_model(args, env):

    log_dir = "./tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    model = create_model(args, env)
    model.learn(total_timesteps=args.total_timesteps)
    model.save(args.model_name)
    plot_results(log_dir, args.model_name)
    return model

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_results(log_folder,title):
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig("sim2real/plots/" + title + ".png")


def main():
    # Create the Hopper environment
    env = gym.make(args.env)


    if args.test is None:
        model = create_model(args, env)
        model = train_model(args, env)
    else:
        None
        # Test here 

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="CustomHopper-source-v0", help="Environment to use")
    parser.add_argument("--total_timesteps", type=int, default=5000, help="The total number of samples to train on")
    parser.add_argument("--model_name", type=str, default="sac_hopper", help="Name of the model to save")
   # parser.add_argument("--render_test", action='store_true', help="Render test")
   # parser.add_argument('--seed', default=0, type=int, help='Random seed')
  #  parser.add_argument('--algo', default='ppo', type=str, help='RL Algo [ppo, sac]')
    #  parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate')
   # parser.add_argument('--gradient_steps', default=-1, type=int, help='Number of gradient steps when policy is updated in sb3 using SAC. -1 means as many as --args.now')
   # parser.add_argument('--test_episodes', default=100, type=int, help='# episodes for test evaluations')
    args = parser.parse_args()
    main()