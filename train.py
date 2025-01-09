import gym
import argparse
import matplotlib.pyplot as plt
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy


def main(model_name, plot_name, num_timesteps):
    # Create the Hopper environment
    env = gym.make('CustomHopper-source-v0')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

    # Initialize the SAC model
    model = SAC('MlpPolicy', env, verbose=1)

    # Train the model
    model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False)

    # Collect rewards
    rewards = []
    for i in range(1, num_timesteps + 1):
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)
        rewards.append(mean_reward)
        print(f"Mean reward after {i} timesteps: {mean_reward}")

    # Save the model
    model.save("sim2real/model/" + model_name)

    # Plot the rewards
    plt.plot(range(1, num_timesteps + 1), rewards)
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title('Training Progress')
    plt.savefig("sim2real/plots/" + plot_name)
    plt.show()

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
    parser.add_argument('--plot_name', type=str, default='plot_name.png', help='The name to save the training plot.')
    parser.add_argument('--num_timesteps', type=int, default=10000, help='The number of timesteps for training.')
    args = parser.parse_args()
    main(args.model_name, args.plot_name, args.num_timesteps)