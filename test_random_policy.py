"""Test a random policy on the Gym Hopper environment

    Play around with this code to get familiar with the
    Hopper environment.

    For example, what happens if you don't reset the environment
    even after the episode is over?
    When exactly is the episode over?
    What is an action here?
"""
import gym
from env.custom_hopper import *
from pyvirtualdisplay import Display
import imageio



def main():

    # Start virtual display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    render = True

    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

    n_episodes = 1
    frames = []


    for ep in range(n_episodes):  
        done = False
        state = env.reset()  # Reset environment to initial state

        while not done:  # Until the episode is over
            action = env.action_space.sample()  # Sample random action

            state, reward, done, info = env.step(action)  # Step the simulator to the next timestep

            if render:
                frame = env.render(mode='rgb_array')
                frames.append(frame)

    video_path = 'sim2real/plots/random_policy.mp4'
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Video saved at {video_path}")
    
    display.stop()


if __name__ == '__main__':
    main()