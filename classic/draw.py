from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from grid import State, GridWorld

ARROWS = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]


def draw_grid(grid: List[List[State]]):
    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots()
    fig.tight_layout()
    grid_image = [[s.color for s in row] for row in grid]
    ax.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
    )  # labels along the bottom edge are off
    # Labels for major ticks
    ax.set_xticklabels(np.arange(0, 7, 1))
    ax.set_yticklabels(np.arange(0, 7, 1))
    # Minor ticks
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.imshow(grid_image)
    return fig, ax


def draw_values(world: GridWorld, value_function: np.ndarray):
    fig, ax = draw_grid(world.grid)
    height, width = world.shape
    cmap = cm.get_cmap("RdYlGn")
    # value_max = np.max(value_function)
    # value_min = np.min(value_function)
    value_max = 10
    value_min = -10
    value_function_grid = np.zeros((width, height, 4))
    for state_idx, state_loc in enumerate(world.valid_locations):
        value = value_function[state_idx]
        x, y = state_loc
        if value == 0.0:
            continue
        ax.text(x, y, f"{value:.2f}", ha="center", va="center")
        value_function_grid[y, x] = cmap((value - value_min) / (value_max - value_min))

    ax.imshow(value_function_grid)
    return ax


def draw_policy(world: GridWorld, policy: np.ndarray):
    fig, ax = draw_grid(world.grid)

    for state_idx, state_loc in enumerate(world.valid_locations):
        state = world.valid_states[state_idx]
        if state.is_absorbing:
            continue

        action = np.argmax(policy[state_idx]).item()
        action_arrow = ARROWS[action]  # Take the corresponding action

        x, y = state_loc
        ax.text(x, y, action_arrow, ha="center", va="center")

    return ax


def draw_episode(world: GridWorld, episodes: List[Tuple[int, int]]):
    fig, ax = draw_grid(world.grid)

    for state_idx, action in episodes:
        action_arrow = ARROWS[action]

        state_loc = world.valid_locations[state_idx]
        x, y = state_loc
        ax.text(x, y, action_arrow, ha="center", va="center")


def draw_rewards(total_rewards: np.ndarray, color=None):
    mean_rewards = np.mean(total_rewards, axis=0)
    std_rewards = np.std(total_rewards, axis=0)

    plt.xlabel("Episode number")
    plt.ylabel("Total reward")
    plt.xlim([0, 300])
    plt.ylim([-120, 50])
    plt.plot(mean_rewards, color=color)
    plt.axhline(y=0.0, linestyle="-", color="black", linewidth=0.5)
    plt.fill_between(
        range(len(mean_rewards)),
        mean_rewards + std_rewards,
        mean_rewards - std_rewards,
        alpha=0.6,
        color=color,
    )
