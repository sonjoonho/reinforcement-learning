from typing import Tuple, Callable

import numpy as np

from grid import GridWorld, State
from util import greedy_policy, generate_epsilon_soft_policy, \
    set_epsilon_greedy_row, compute_return


def q_learning(world: GridWorld,
               discount: float,
               n_episodes: int,
               epsilon: float,
               learning_rate: float,
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_states = len(world.valid_states)
    n_actions = 4

    Q: np.ndarray = np.random.rand(n_states, n_actions)
    filter_fn: Callable[[State], bool] = lambda s: s.is_absorbing
    terminal_states = [filter_fn(s) for s in world.valid_states]
    Q[terminal_states, :] = 0

    policy: np.ndarray = np.zeros_like(Q)

    intermediate_policies = np.zeros((n_episodes, n_states, n_actions))

    # Track our rewards for each episodes for plotting.
    total_rewards = np.zeros(n_episodes)

    for i in range(n_episodes):
        # Track our rewards for each episodes for plotting. Note that this
        # needs to be an appendable list since we don't know the episode length
        # in advance.
        rewards = []

        cur_state_idx: int = world.generate_starting_state()
        while not world.valid_states[cur_state_idx].is_absorbing:
            # Set the probability in-place for each action in an epsilon-greedy
            # way.
            set_epsilon_greedy_row(policy, Q, cur_state_idx, n_actions, epsilon)

            new_state_idx, action = world.generate_episode_step(policy,
                                                                cur_state_idx)
            immediate_reward = world.valid_states[new_state_idx].reward
            rewards.append(immediate_reward)

            Q[cur_state_idx, action] += learning_rate * (
                    immediate_reward + discount * np.max(Q[new_state_idx]) - Q[
                cur_state_idx, action])
            cur_state_idx = new_state_idx

        total_rewards[i] = compute_return(np.asarray(rewards), discount)
        intermediate_policies[i] = policy.copy()

    return greedy_policy(policy), Q, total_rewards, intermediate_policies


def monte_carlo(world: GridWorld,
                discount: float,
                n_episodes: int,
                epsilon: float,
                learning_rate: float,
                logging: bool = False,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Implementation of the incremental, first-visit Monte-Carlo control
    algorithm.
    """
    n_states = len(world.valid_states)
    n_actions = 4

    # State-action values.
    Q: np.ndarray = np.zeros((n_states, n_actions))

    # Our policy to be iteratively optimised.
    policy: np.ndarray = generate_epsilon_soft_policy(n_states, n_actions,
                                                      epsilon)

    intermediate_policies = np.zeros((n_episodes, n_states, n_actions))

    # Track our rewards for each episodes for plotting.
    total_rewards = np.zeros(n_episodes)

    for i in range(n_episodes):
        if logging:
            print(f'Running episode {i + 1}/{n_episodes}')
        episode = world.generate_episode(policy)
        episode_states = [idx for (idx, _) in episode]

        # Track rewards for this episode.
        rewards = np.zeros(len(episode) - 1)

        # The return from the first appearance of this state-action pair in the
        # episode.
        cur_return: float = 0.0
        # Loop over each episode from t = T-1, ..., 0
        for t, (state_idx, action_taken) in reversed(list(enumerate(episode[:-1]))):
            immediate_reward = world.valid_states[episode_states[t + 1]].reward
            rewards[t] = immediate_reward

            # Get the return for this state.
            cur_return = discount * cur_return + immediate_reward

            # We are only interested in the first appearance, so skip if it
            # appears earlier in the episode.
            if state_idx in episode_states[:t]:
                continue

            Q[state_idx, action_taken] += learning_rate * (
                    cur_return - Q[state_idx, action_taken])

            # Set the probability in-place for each action in an epsilon-greedy
            # way.
            set_epsilon_greedy_row(policy, Q, state_idx, n_actions, epsilon)

        total_rewards[i] = compute_return(rewards, discount)
        intermediate_policies[i] = policy.copy()

    # Quantise our epsilon-soft policy.
    return greedy_policy(policy), Q, total_rewards, intermediate_policies


def value_iteration(world: GridWorld,
                    threshold: float,
                    discount: float
                    ) -> np.ndarray:
    delta = threshold

    n_states, n_actions = len(world.valid_states), 4

    V: np.ndarray = np.zeros(n_states)

    T: np.ndarray = world.transition_matrix
    R: np.ndarray = world.reward_matrix

    while delta >= threshold:
        delta = 0.0

        for state_idx, state in enumerate(world.valid_states):
            if state.is_absorbing:
                continue

            v = V[state_idx]
            # Compute Q for each action from this state.
            state_Q = np.zeros(n_actions)
            for next_state_idx in range(n_states):
                state_Q += T[next_state_idx, state_idx, :] * (
                        R[next_state_idx, state_idx, :] + discount * V[
                    next_state_idx])
            V[state_idx] = np.max(state_Q)
            delta = max(delta, np.abs(v - V[state_idx]))

    policy = np.zeros((n_states, n_actions))
    policy[:, 0] = 1

    for state_idx, state in enumerate(world.valid_states):
        if state.is_absorbing:
            continue

        # Compute Q for each action from this state.
        state_Q = np.zeros(n_actions)
        for next_state_idx in range(n_states):
            state_Q += T[next_state_idx, state_idx, :] * (
                    R[next_state_idx, state_idx, :] + discount * V[
                next_state_idx])

        # Create a new policy for this state.
        new_policy = np.zeros(n_actions)
        new_action = np.argmax(state_Q)
        new_policy[new_action] = 1
        policy[state_idx] = new_policy

    return policy


def policy_iteration(world: GridWorld, threshold: float,
                     discount: float) -> np.ndarray:
    n_states, n_actions = len(world.valid_states), 4

    T: np.ndarray = world.transition_matrix
    R: np.ndarray = world.reward_matrix

    # Initialise the policy to always choose action 1.
    policy: np.ndarray = np.zeros((n_states, n_actions))
    policy[:, 0] = 1

    policy_stable = False
    while not policy_stable:
        V = policy_evaluation(world, policy, threshold, discount)
        policy_stable = True

        for state_idx, state in enumerate(world.valid_states):
            if state.is_absorbing:
                continue

            old_action = np.argmax(policy[state_idx, :])

            # Compute Q for each action from this state.
            state_Q = np.zeros(n_actions)
            for next_state_idx in range(n_states):
                state_Q += T[next_state_idx, state_idx, :] * (
                        R[next_state_idx, state_idx, :] + discount * V[
                    next_state_idx])

            # Create a new policy for this state.
            new_policy = np.zeros(n_actions)
            new_action = np.argmax(state_Q)
            new_policy[new_action] = 1
            policy[state_idx] = new_policy

            # We have converged if the policy does not change.
            if old_action != new_action:
                policy_stable = False

    return policy


def policy_evaluation(world: GridWorld, policy: np.ndarray, threshold: float,
                      discount: float) -> np.ndarray:
    delta: float = 2 * threshold

    n_states, n_actions = policy.shape

    V: np.ndarray = np.zeros(n_states)
    V_new: np.ndarray = np.copy(V)

    T: np.ndarray = world.transition_matrix
    R: np.ndarray = world.reward_matrix

    while delta > threshold:
        for state_idx, state in enumerate(world.valid_states):
            if state.is_absorbing:
                continue

            state_V: float = 0
            for action_idx in range(n_actions):
                state_Q: float = 0
                for next_state_idx in range(n_states):
                    state_Q += T[next_state_idx, state_idx, action_idx] * (
                            R[
                                next_state_idx, state_idx, action_idx] + discount *
                            V[next_state_idx])
                state_V += policy[state_idx, action_idx] * state_Q

            V_new[state_idx] = state_V

        delta = max(abs(V_new - V))
        V = np.copy(V_new)

    return V
