import socket
import time

import numpy as np
from sklearn.model_selection import ParameterGrid

from agent import Agent
from draw import plot_trace
from random_environment import Environment


def reward_1(next_state: np.array, distance_to_goal: float):
    reward: float = - distance_to_goal
    if distance_to_goal < 0.03:
        reward = 1.0
    return reward


def main():
    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.

    seeds = [391]
    discounts = [0.99]
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    target_update_intervals = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    learning_rate = [0.00025]
    clips = [10]
    alphas = [0.2]
    betas = [0]
    replay_epsilons = [0.1]
    n_actions = [3]
    episode_lengths = [850]
    episode_length_decays = [1.0]
    epsilon_decays = [0.97]
    epsilon_mins = [0.0001]
    rewards = [reward_1]
    weight_decays = [0.001]

    param_grid = {
        'discount': discounts,
        'batch_size': batch_sizes,
        'target_update_interval': target_update_intervals,
        'learning_rate': learning_rate,
        'clip': clips,
        'alpha': alphas,
        'beta': betas,
        'replay_epsilon': replay_epsilons,
        'n_action': n_actions,
        'episode_length': episode_lengths,
        'episode_length_decay': episode_length_decays,
        'seed': seeds,
        'epsilon_decay': epsilon_decays,
        'reward': rewards,
        'epsilon_min': epsilon_mins,
        'weight_decay': weight_decays,
    }
    grid = ParameterGrid(param_grid)
    print(f'Running {len(grid)} parameters')

    # segments texel41 - 0:18, texel42 - 18:36, texel43 - 36:54, texel44 - 54:72
    texelmap = {
        'texel39': (0, 49),
        'texel43': (49, 98),
        'texel41': (98, 147),
        'texel44': (147, 196),
    }
    hostname = socket.gethostname().split('.')[0]
    START, END = texelmap[hostname]
    print(f'Running {hostname} range {START}:{END}')
    for i, params in enumerate(list(grid)[START:END], START):
        run(str(i), params)


def run(index: str, params):
    agent = Agent(
        logging=False,
        discount=params['discount'],
        batch_size=params['batch_size'],
        target_update_interval=params['target_update_interval'],
        learning_rate=params['learning_rate'],
        clip=params['clip'],
        replay_alpha=params['alpha'],
        replay_beta=params['beta'],
        replay_epsilon=params['replay_epsilon'],
        n_actions=params['n_action'],
        episode_length=params['episode_length'],
        episode_length_decay=params['episode_length_decay'],
        epsilon_decay=params['epsilon_decay'],
        epsilon_min=params['epsilon_min'],
        reward_function=params['reward'],
        weight_decay=params['weight_decay'],
    )
    random_seed = params['seed']
    np.random.seed(random_seed)
    # Create a random environment
    environment = Environment(magnification=500)

    # Get the initial state
    state = environment.init_state

    # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
    start_time = time.time()
    end_time = start_time + 600

    # Train the agent, until the time is up
    while time.time() < end_time:
        # If the action is to start a new episode, then reset the state
        if agent.has_finished_episode():
            state = environment.init_state
        # Get the state and action from the agent
        action = agent.get_next_action(state)
        # Get the next state and the distance to the goal
        next_state, distance_to_goal = environment.step(state, action)
        # Return this to the agent
        agent.set_next_state_and_distance(next_state, distance_to_goal)
        # Set what the new state is
        state = next_state
        # Optionally, show the environment

    # plot_loss(agent, index)
    # plot_reward(agent, index)
    # visualise(agent, index)
    print(f'Agent reached the goal {agent.reached_goal_times} times.')

    # Test the agent for 100 steps, using its greedy policy
    state = environment.init_state
    trace = [state]
    has_reached_goal = False
    for step_num in range(100):
        action = agent.get_greedy_action(state)
        next_state, distance_to_goal = environment.step(state, action)
        trace.append(next_state)
        # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
        if distance_to_goal < 0.03:
            has_reached_goal = True
            break
        state = next_state

    environment.draw(trace[-1])
    plot_trace(trace, environment)
    environment.save(index)

    # Print out the result
    if has_reached_goal:
        result_str = f'{index} - Reached goal in ' + str(step_num) + ' steps.'
    else:
        result_str = f'{index} - Did not reach goal. Final distance = ' + str(distance_to_goal)

    print(result_str)

    with open(f'runs/run_{index}.txt', 'w') as f:
        f.write(f'{result_str} {str(params)}')


if __name__ == "__main__":
    main()
