import random
from enum import Enum
from typing import List, Tuple, Set

import numpy as np
from dataclasses import dataclass

DEFAULT_REWARD = -1
PENALTY_REWARD = -100
REWARD_REWARD = 10

RGB_WHITE = (255, 255, 255)
RGB_BLACK = (0, 0, 0)
RGB_RED = (151, 0, 37)
RGB_GREEN = (0, 97, 41)


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


@dataclass
class State:
    is_wall: bool = False
    is_absorbing: bool = False
    reward: int = 0
    color: Tuple[int, int, int] = RGB_WHITE


def d() -> State:
    """Default state."""
    return State(reward=DEFAULT_REWARD)


def w() -> State:
    """Walls."""
    return State(is_wall=True, color=RGB_BLACK)


def p() -> State:
    """Penalty state."""
    return State(reward=PENALTY_REWARD, color=RGB_RED, is_absorbing=True)


def r() -> State:
    """Reward state."""
    return State(reward=REWARD_REWARD, color=RGB_GREEN, is_absorbing=True)


class GridWorld:
    grid: List[List[State]]

    def __init__(self, grid: List[List[State]], action_probability=1.0):
        assert all(len(row) == len(grid[0]) for row in grid)
        self.grid = grid
        self.shape = (len(grid), len(grid[0]))

        self.valid_locations, self.valid_states = self._compute_valid_locations()

        self.walls = self._compute_walls()

        # action_probability is the probability of the desired action succeeding. If it fails, we move in one of the
        # other directions with equal probability.
        self.action_effect_probabilities: List[float] = [action_probability] + [
            (1.0 - action_probability) / 3.0
        ] * 3

        self.transition_matrix, self.reward_matrix = self._compute_transition_matrix()

    def _compute_valid_locations(self) -> Tuple[List[Tuple[int, int]], List[State]]:
        valid_locations: List[Tuple[int, int]] = []
        valid_states: List[State] = []
        for y, row in enumerate(self.grid):
            for x, s in enumerate(row):
                if not s.is_wall:
                    valid_locations.append((x, y))
                    valid_states.append(s)

        return valid_locations, valid_states

    def _compute_walls(self) -> Set[Tuple[int, int]]:
        walls = set()
        for y, row in enumerate(self.grid):
            for x, s in enumerate(row):
                if s.is_wall:
                    walls.add((x, y))

        return walls

    def _is_valid_location(self, loc: Tuple[int, int]):
        x, y = loc
        height, width = self.shape
        is_wall = loc in self.walls
        is_out_of_bounds = x < 0 or y < 0 or x > width - 1 or y > height - 1
        return not (is_wall or is_out_of_bounds)

    def _compute_next_location(
        self, loc: Tuple[int, int], direction: Direction
    ) -> Tuple[int, int]:
        old_x, old_y = loc
        if direction == Direction.NORTH:
            new_loc = (old_x, old_y - 1)
        elif direction == Direction.EAST:
            new_loc = (old_x + 1, old_y)
        elif direction == Direction.SOUTH:
            new_loc = (old_x, old_y + 1)
        elif direction == Direction.WEST:
            new_loc = (old_x - 1, old_y)
        else:
            raise ValueError("Unknown direction")

        if not self._is_valid_location(new_loc):
            return loc

        return new_loc

    def _compute_transition_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        n_states = len(self.valid_locations)
        transition_matrix = np.zeros((n_states, n_states, 4))

        # There can only be 4 actions.
        for action in range(4):
            # See https://piazza.com/class/kf7uhzvqmwa14k?cid=93.
            for effect in range(4):
                outcome = (action + effect + 1) % 4
                if outcome == 0:
                    outcome = 3
                else:
                    outcome -= 1

                prob = self.action_effect_probabilities[effect]
                for prior_state_number, prior_location in enumerate(
                    self.valid_locations
                ):
                    post_location = self._compute_next_location(
                        prior_location, Direction(outcome)
                    )
                    post_state_number = self.valid_locations.index(post_location)
                    transition_matrix[
                        post_state_number, prior_state_number, action
                    ] += prob

        reward_matrix = DEFAULT_REWARD * np.ones((n_states, n_states, 4))
        for state_idx, state in enumerate(self.valid_states):
            if state.is_absorbing:
                # I don't understand this.
                reward_matrix[state_idx, :, :] = state.reward

        return transition_matrix, reward_matrix

    def generate_starting_state(self) -> int:
        n_states = len(self.valid_states)
        cur_state_idx: int = random.randint(0, n_states - 1)
        while self.valid_states[cur_state_idx].is_absorbing:
            cur_state_idx = random.randint(0, n_states - 1)
        return cur_state_idx

    def generate_episode_step(
        self, policy: np.ndarray, cur_state_idx: int
    ) -> Tuple[int, int]:
        n_states = len(self.valid_states)
        n_actions = 4

        cur_loc = self.valid_locations[cur_state_idx]

        desired_action: int = random.choices(
            range(n_actions), weights=policy[cur_state_idx]
        )[0]
        # After choosing an action as dictated by the policy, we use the transition matrix to determine the actual
        # next state is.
        transition_probabilities = self.transition_matrix[
            :, cur_state_idx, desired_action
        ]
        new_state_idx: int = random.choices(
            range(n_states), weights=transition_probabilities
        )[0]
        new_loc = self.valid_locations[new_state_idx]

        # From the actual movement, we can infer the actual action taken.
        actual_action: int = self._compute_actual_action(
            Direction(desired_action), cur_loc, new_loc
        ).value

        return new_state_idx, actual_action

    def generate_episode(self, policy: np.ndarray) -> List[Tuple[int, int]]:
        """Returns a list of (state, action) pairs as tuples of integers."""
        n_states = len(self.valid_states)
        n_actions = 4
        assert policy.shape == (n_states, n_actions)

        episode: List[(State, Direction)] = []

        cur_state_idx: int = self.generate_starting_state()
        while not self.valid_states[cur_state_idx].is_absorbing:
            new_state_idx, action = self.generate_episode_step(policy, cur_state_idx)

            episode.append((cur_state_idx, action))
            cur_state_idx = new_state_idx

        # The action doesn't matter for the final state.
        episode.append((cur_state_idx, 0))
        return episode

    @staticmethod
    def _compute_actual_action(
        desired_action: Direction, cur_loc: Tuple[int, int], new_loc: Tuple[int, int]
    ) -> Direction:
        """Computes the action taken from two locations. If there is no movement, it default to the desired action."""
        cur_x, cur_y = cur_loc
        new_x, new_y = new_loc
        delta: Tuple[int, int] = (cur_x - new_x, cur_y - new_y)

        if delta == (0, -1):
            return Direction.SOUTH
        elif delta == (0, 1):
            return Direction.NORTH
        elif delta == (1, 0):
            return Direction.WEST
        elif delta == (-1, 0):
            return Direction.EAST
        elif delta == (0, 0):
            return desired_action
        else:
            raise ValueError(f"Unknown delta: {delta}")
