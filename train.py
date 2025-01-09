"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

def main():

    # Create the Hopper environment
    env = gym.make('CustomHopper-source-v0')


    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

    # Initialize the SAC model
    model = SAC('MlpPolicy', env, verbose=1)

    # Train the model
    model.learn(total_timesteps=100000)

    # Save the model
    model.save("sac_hopper")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Test the policy on the target environment
    target_env = gym.make('CustomHopper-target-v0')
    mean_reward, std_reward = evaluate_policy(model, target_env, n_eval_episodes=10)
    print(f"Mean reward on target environment: {mean_reward} +/- {std_reward}")

if __name__ == '__main__':
    main()