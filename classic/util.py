from typing import List, Tuple

import numpy as np

from grid import GridWorld


def greedy_policy(Q: np.ndarray) -> np.ndarray:
    """Creates an greedy policy from Q."""
    n_states, n_actions = Q.shape
    policy: np.ndarray = np.zeros_like(Q)
    # Set the max element in each row to 1, all others to 0.
    policy[np.arange(n_states), Q.argmax(1)] = 1
    return policy


def generate_epsilon_soft_policy(n_states: int, n_actions: int, epsilon: float):
    """Generates an arbitrary epsilon-soft policy."""
    big_p = 1 - epsilon + epsilon / n_actions
    small_p = epsilon / n_actions
    row: np.ndarray = np.asarray([big_p] + [small_p for _ in range(n_actions - 1)])
    assert row.shape == (n_actions,)
    policy: np.ndarray = np.tile(row, (n_states, 1))
    assert policy.shape == (n_states, n_actions)
    list(map(np.random.shuffle, policy))
    return policy


def compute_value_function(Q: np.ndarray, policy: np.ndarray) -> np.ndarray:
    assert Q.shape == policy.shape
    n_states, _ = Q.shape
    V: np.ndarray = np.sum(Q * policy, axis=1)
    assert V.shape == (n_states,)
    return V


def compute_return(rewards: np.ndarray, discount: float) -> float:
    """Performs backward-discounting."""
    exponents = range(len(rewards))[::-1]
    discounts = np.ones(len(rewards)) * discount
    return np.sum(np.power(discounts, exponents) * rewards).item()
    # return np.mean(rewards).item()


def evaluate_policy(world: GridWorld, policy: np.ndarray, n_episodes: int) -> float:
    returns = np.zeros(n_episodes)
    for i in range(n_episodes):
        episode = world.generate_episode(policy)
        returns[i] = np.sum(get_rewards(world, episode))
    return np.mean(returns).item()


def set_epsilon_greedy_row(policy: np.ndarray, Q: np.ndarray, cur_state_idx: int, n_actions: int,
                           epsilon: float) -> None:
    optimal_action: int = np.argmax(Q[cur_state_idx, :]).item()
    for action in range(n_actions):
        if action == optimal_action:
            prob = 1 - epsilon + epsilon / n_actions
        else:
            prob = epsilon / n_actions
        policy[cur_state_idx, action] = prob


def get_rewards(world: GridWorld, episode: List[Tuple[int, int]]) -> np.ndarray:
    return np.asarray([world.valid_states[state_idx].reward for (state_idx, _) in episode[1:]])
