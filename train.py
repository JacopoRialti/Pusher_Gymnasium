import gym
import argparse
import matplotlib.pyplot as plt
from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback # Callback class for logging rewards


class reward_callback(BaseCallback):
    def __init__(self, verbose=0):
        super(reward_callback, self).__init__(verbose)
        self.rewards = []
        
    def _on_step(self) -> bool:
        reward = self.model.rewards
        self.rewards.append(reward)
        return True

    def get_rewards(self):
        return self.rewards


# ottieni informazioni sulla reward
def reward_info(model, env, episodes=10):
    total_reward = 0
    num_steps = 0

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        num_steps += 1
        print(f"Step: {i}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            break

    mean_reward = total_reward / num_steps if num_steps > 0 else 0
    print(f"Mean reward: {mean_reward}")
    return mean_reward


def plot_rewards(rewards, plot_name):
    plt.plot(rewards)
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.savefig("sim2real/plots/" + plot_name)
    plt.show()
    

def main(model_name, plot_name, num_timesteps):
    # Create the Hopper environment
    env = gym.make('CustomHopper-source-v0')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

    # Initialize the SAC model
    model = SAC('MlpPolicy', env, verbose=1)

    # Train the model
    callback_ep = reward_callback()
    
    model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False, callback_ep = callback_ep)
    
    # Save the model
    model.save("sim2real/models/" + model_name)

    #episode_rewards = callback.get_rewards()
    plot_rewards(callback_ep.get_rewards(), plot_name)

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    '''
    # Test the policy on the target environment
    target_env = gym.make('CustomHopper-target-v0')
    mean_reward, std_reward = evaluate_policy(model, target_env, n_eval_episodes=10)
    print(f"Mean reward on target environment: {mean_reward} +/- {std_reward}") '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a SAC model on the CustomHopper environment.')
    parser.add_argument('--model_name', type=str, default='model_name', help='The name to save the trained model.')
    parser.add_argument('--plot_name', type=str, default='plot_name.png', help='The name to save the training plot.')
    parser.add_argument('--num_timesteps', type=int, default=10000, help='The number of timesteps for training.')
    args = parser.parse_args()
    main(args.model_name, args.plot_name, args.num_timesteps)