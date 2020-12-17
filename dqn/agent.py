############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################
import time
from typing import NamedTuple, Tuple, List, Callable

import numpy as np
import torch

# DQN parameters.

BATCH_SIZE = 128
N_ACTIONS = 3
DISCOUNT = 0.99
TARGET_NETWORK_UPDATE_INTERVAL = 2

INITIAL_EPISODE_LENGTH = 850
MIN_EPISODE_LENGTH = 100
EPISODE_LENGTH_DECAY = 1.0

INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.00001
EPSILON_DECAY = 0.96
EPSILON_GROWTH = 1.0003

# Network parameters.

CLIP = 10
LEARNING_RATE = 0.00025
WEIGHT_DECAY = 0.001

# Replay buffer parameters.

REPLAY_ALPHA = 0.2

REPLAY_BETA = 0.0
REPLAY_BETA_GROWTH = 0.0005

REPLAY_EPSILON = 0.1

BUFFER_CAPACITY = 500000

EVALUATE_INTERVAL = 1


def default_reward(next_state: np.array, distance_to_goal: float):
    reward: float = -distance_to_goal
    if distance_to_goal < 0.03:
        reward = 1.0
    return reward


class Batch(NamedTuple):
    """
    Batch is a data class representing a minibatch of transitions.
    """

    # states is a 2D array of states.
    states: np.ndarray
    # actions is a 1D array of discrete actions.
    actions: np.ndarray
    # rewards is a 1D array of rewards.
    rewards: np.ndarray
    # next_state is a 2D array of states.
    next_states: np.ndarray


class ReplayBuffer:
    """
    Prioritised replay buffer as outlined in https://arxiv.org/pdf/1511.05952.pdf.
    """

    def __init__(
        self,
        capacity: int = BUFFER_CAPACITY,
        epsilon: float = REPLAY_EPSILON,
        alpha: float = REPLAY_ALPHA,
        beta: float = REPLAY_BETA,
        beta_growth: float = REPLAY_BETA_GROWTH,
    ):
        self._size: int = 0
        self._capacity: int = capacity
        self._cursor: int = 0
        self._alpha: float = alpha
        self._beta: float = beta
        # beta_growth is the rate at which we *linearly* anneal beta.
        self._beta_growth: float = beta_growth

        # epsilon is a small constant added to the weight of all transitions.
        self._epsilon = epsilon

        self._states = np.empty((capacity, 2), dtype=np.float32)
        self._actions = np.empty((capacity, 1), dtype=np.int)
        self._rewards = np.empty((capacity, 1), dtype=np.float32)
        self._next_states = np.empty((capacity, 2), dtype=np.float32)
        self._priorities = np.empty((capacity,), dtype=np.float32)

        # max_priority is the largest weight we've seen during training.
        self._max_priority: float = 0.1
        self._rng = np.RandomState = np.random.RandomState(int(time.time()))

    def put(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray
    ):
        assert state.shape == (2,)
        assert next_state.shape == (2,)

        c = self._cursor

        self._states[c] = state
        self._actions[c] = action
        self._rewards[c] = reward
        self._next_states[c] = next_state
        self._priorities[c] = self._max_priority

        self._size += 1
        self._cursor = self._size % self._capacity

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray, np.ndarray]:
        """
        Returns a sample from the replay buffer of size batch_size. Transitions are sampled according to their
        priorities, and are adjusted for bias using Importance Sampling.
        """
        assert self.size >= batch_size

        # length is how much of the numpy array we've used so far.
        length = min(self._capacity, self._size)

        priorities = self._priorities[:length]
        p = priorities / priorities.sum()
        sample_idx: np.ndarray = self._rng.choice(
            length, batch_size, p=p, replace=False
        )

        # Note that np.empty() returns uninitialised memory so we must ensure that each entry is properly initialised.
        states = self._states[sample_idx]
        actions = self._actions[sample_idx]
        rewards = self._rewards[sample_idx]
        next_states = self._next_states[sample_idx]
        batch: Batch = Batch(states, actions, rewards, next_states)

        # Compute importance-sampling weights.
        weights = (self._capacity * p[sample_idx]) ** (-self._beta)
        # Normalise by the largest weight in the batch.
        weights /= weights.max()

        # Anneal beta every 100 samples to avoid doing the computation on each step. We can use self.size for this
        # since it is incremented on each step.
        if self._size % 100 == 0:
            self._beta = min(self._beta + self._beta_growth, 1.0)

        assert sample_idx.shape == (batch_size,)
        assert weights.shape == sample_idx.shape

        return batch, sample_idx, weights

    def update(self, batch_idx: np.ndarray, errors: np.ndarray):
        """
        Update the priorities of the specified transitions.
        """
        assert batch_idx.shape == errors.shape
        self._priorities[batch_idx] = (np.abs(errors) + self._epsilon) ** self._alpha
        self._max_priority = max(self._max_priority, self._priorities.max())

    def __len__(self) -> int:
        return min(self._size, self._capacity)


class Agent:
    action_map: List[np.ndarray] = [
        np.array([0.02, 0.0]),  # East
        np.array([0, 0.02]),  # North
        np.array([0, -0.02]),  # South
    ]

    def __init__(
        self,
        epsilon: float = INITIAL_EPSILON,
        epsilon_min: float = MIN_EPSILON,
        epsilon_decay: float = EPSILON_DECAY,
        epsilon_growth: float = EPSILON_GROWTH,
        episode_length: int = INITIAL_EPISODE_LENGTH,
        episode_length_decay: float = EPISODE_LENGTH_DECAY,
        batch_size: int = BATCH_SIZE,
        discount: float = DISCOUNT,
        learning_rate: float = LEARNING_RATE,
        target_update_interval: int = TARGET_NETWORK_UPDATE_INTERVAL,
        clip: float = CLIP,
        n_actions: int = N_ACTIONS,
        replay_alpha: float = REPLAY_ALPHA,
        replay_beta: float = REPLAY_BETA,
        replay_epsilon: float = REPLAY_EPSILON,
        reward_function: Callable[[np.ndarray, float], float] = default_reward,
        weight_decay: float = WEIGHT_DECAY,
    ):
        # We use our own RNG which isn't effected by the global seed.
        self.rng = np.random.RandomState(int(time.time()))

        # Mini-batch size.
        self.batch_size: int = batch_size
        # The number of actions. Used to control whether or not diagonal actions are used.
        self.n_actions = n_actions
        self.reward_function = reward_function
        self.discount: float = discount
        self.learning_rate: float = learning_rate
        self.target_update_interval: int = target_update_interval

        # This group of variables are used to track the value of epsilon (exploration parameter). We decay epsilon
        # between episodes, but grow epsilon within each episode.
        self.episode_epsilon: float = epsilon
        self.current_epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_growth: float = epsilon_growth

        # Truncate episode_length in case anything other than an int has been passed.
        self.episode_length: int = int(episode_length)
        self.episode_length_decay: float = episode_length_decay

        # Capture information about the latest state of the agent.
        self.num_steps_taken: int = 0
        self.state: np.ndarray = np.array([0, 0])
        self.action: int = 0

        self.episodes_finished: int = 0
        self.num_steps_taken: int = 0
        self.steps_in_episode: int = 0

        self.dqn: DQN = DQN(
            batch_size=batch_size,
            discount=discount,
            learning_rate=learning_rate,
            clip=clip,
            n_actions=n_actions,
            weight_decay=weight_decay,
        )
        self.buffer: ReplayBuffer = ReplayBuffer(
            alpha=replay_alpha, beta=replay_beta, epsilon=replay_epsilon
        )

        # Tracks when we are in 'evaluation mode'.
        self.evaluating = False
        self.stop_training = False
        self.evaluating_steps = 0

    def _discrete_to_continuous_action(self, action: int) -> np.ndarray:
        """
        Converts a discrete action to a continuous actions.
        """
        return self.action_map[action]

    def has_finished_episode(self):
        """
        Returns whether or not we have finished the current episode. Also runs various hooks that occur at the end
        of each episode.
        """
        if self.num_steps_taken == 0:
            return False

        if self.evaluating:
            return False

        done = self.num_steps_taken % self.episode_length == 0
        if done:
            self.finished_episode_tasks()

        # We evaluate our greedy policy every EVALUATE_INTERVAL episodes.
        evaluation_episode = (
            done
            and self.episodes_finished >= 80
            and self.episodes_finished % EVALUATE_INTERVAL == 0
        )
        if evaluation_episode:
            self.num_steps_taken += 1
            self.evaluating = True

        return done

    def get_next_action(self, state: np.ndarray) -> np.ndarray:
        """
        Returns the next *continuous* action.
        """
        assert state.shape == (2,)

        # If we are in evaluation mode, we simply return the next greedy action. Otherwise, we choose the next action
        # epsilon-greedily based on the output of the Q-network.
        if self.evaluating:
            self.state = state
            self.evaluating_steps += 1
            return self.get_greedy_action(state)

        pred: np.ndarray = self.dqn.predict(state)
        optimal_action: int = np.argmax(pred).item()
        p = self.rng.random()
        if p > self.current_epsilon:
            action = optimal_action
        else:
            action = self.rng.randint(self.n_actions)

        self.num_steps_taken += 1
        self.steps_in_episode += 1
        self.state = state
        # This is only used to add the transition into the buffer, so no need to set it when in evaluation mode.
        self.action = action

        return self._discrete_to_continuous_action(action)

    def set_next_state_and_distance(
        self, next_state: Tuple[float, float], distance_to_goal: float
    ):
        """
        Performs a single training update.
        """
        if self.stop_training:
            return

        # Note that the most recent state and action are set in get_next_action.
        next_state = np.asarray(next_state)
        assert next_state.shape == (2,)

        reward = self.reward_function(next_state, distance_to_goal)

        # Grow episode within each episode.
        self.current_epsilon = min(self.current_epsilon * self.epsilon_growth, 1.0)

        self.buffer.put(self.state, self.action, reward, next_state)

        # Skip training if the buffer isn't big enough yet.
        if len(self.buffer) < self.batch_size:
            return

        if self.evaluating:
            reached_goal = distance_to_goal < 0.03
            finished_evaluation = self.evaluating_steps >= 100

            if reached_goal:
                # We stop training if we have found a greedy policy that works.
                self.stop_training = True

            if finished_evaluation or reached_goal:
                self.evaluating = False
                self.evaluating_steps = 0

            return

        batch, batch_idx, weights = self.buffer.sample(self.batch_size)

        loss, td_error = self.dqn.train_q_network(batch, weights)
        self.buffer.update(batch_idx, td_error)

        if self.num_steps_taken % self.target_update_interval == 0:
            self.dqn.update_target_network()

    def get_greedy_action(self, state: np.ndarray) -> np.ndarray:
        """
        Returns the next greedy *continuous* action.
        """
        assert state.shape == (2,)

        pred: np.ndarray = self.dqn.predict(state)
        optimal_action: int = np.argmax(pred).item()

        action: np.ndarray = self._discrete_to_continuous_action(optimal_action)
        return action

    def finished_episode_tasks(self):
        """
        Bunch of things that happen at the end of each episode.
        """
        self.episodes_finished += 1

        # Decay epsilon and episode length per episode, only once we have reached the goal.
        self.episode_epsilon = max(
            self.episode_epsilon * self.epsilon_decay, self.epsilon_min
        )

        # Reset per-episode variables.
        self.current_epsilon = self.episode_epsilon
        self.steps_in_episode = 0


class Network(torch.nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=256)
        self.layer_2 = torch.nn.Linear(in_features=256, out_features=256)
        self.layer_3 = torch.nn.Linear(in_features=256, out_features=256)
        self.output_layer = torch.nn.Linear(
            in_features=256, out_features=output_dimension
        )

    def forward(self, X: torch.Tensor):
        layer_1_output = torch.nn.functional.relu(self.layer_1(X))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output: torch.Tensor = self.output_layer(layer_3_output)
        return output


class DQN:
    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        discount: float = DISCOUNT,
        learning_rate: float = LEARNING_RATE,
        clip: float = CLIP,
        n_actions: int = N_ACTIONS,
        weight_decay: float = WEIGHT_DECAY,
    ):
        self._clip = clip
        self._n_actions = n_actions
        self._batch_size: int = batch_size
        self._discount: float = discount
        self._learning_rate: float = learning_rate
        self._q_network: Network = Network(
            input_dimension=2, output_dimension=self._n_actions
        )
        self._target_network: Network = Network(
            input_dimension=2, output_dimension=n_actions
        )
        self._target_network.eval()

        # Synchronise policy network and target network.
        self.update_target_network()
        self._optimiser = torch.optim.AdamW(
            self._q_network.parameters(),
            lr=self._learning_rate,
            weight_decay=weight_decay,
        )

    def train_q_network(
        self, batch: Batch, weights: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        td_error = self._calculate_error(batch)

        # Apply importance sampling by weighting the TD error.
        weights = torch.from_numpy(weights).unsqueeze(-1)
        assert weights.shape == td_error.shape

        loss = self._calculate_loss(weights, td_error)
        self._optimiser.zero_grad()
        loss.backward()

        # Perform gradient clipping.
        torch.nn.utils.clip_grad_norm_(self._q_network.parameters(), self._clip)

        self._optimiser.step()

        return loss.item(), td_error.detach().numpy().reshape(-1)

    def _calculate_loss(
        self, weights: torch.Tensor, error: torch.Tensor
    ) -> torch.Tensor:
        # Compute MSE from error.
        loss = (weights * error).square().mean()
        # An empty shape denotes a scalar value.
        assert loss.shape == ()
        return loss

    def _calculate_error(self, batch: Batch) -> torch.Tensor:
        """
        Returns the TD error.
        """
        # numpy and pytorch share the same memory locations, so this is cheap.
        states = torch.from_numpy(batch.states)
        rewards = torch.from_numpy(batch.rewards)
        actions = torch.from_numpy(batch.actions)
        next_states = torch.from_numpy(batch.next_states)

        q_pred_state = self._q_network(states)
        assert q_pred_state.shape == (self.batch_size, self.n_actions)

        q_pred_state_actions = q_pred_state.gather(dim=1, index=actions)
        assert q_pred_state_actions.shape == (self.batch_size, 1)

        target_pred_next_state = self._target_network(next_states)
        assert target_pred_next_state.shape == (self.batch_size, self.n_actions)

        max_actions_target_pred_next_state = torch.argmax(
            target_pred_next_state, dim=1
        ).unsqueeze(-1)
        assert max_actions_target_pred_next_state.shape == (self.batch_size, 1)

        q_pred_next_state = self._q_network(next_states)
        assert q_pred_next_state.shape == (self.batch_size, self.n_actions)

        q_pred_next_state_actions = q_pred_next_state.gather(
            dim=1, index=max_actions_target_pred_next_state
        )
        assert q_pred_next_state_actions.shape == (self.batch_size, 1)

        returns: torch.Tensor = rewards + self._discount * q_pred_next_state_actions
        assert returns.shape == q_pred_state_actions.shape

        # We do not reduce the result yet so we can apply prioritised experience replay with importance sampling.
        td_error = q_pred_state_actions - returns.detach()
        return td_error

    def update_target_network(self):
        """
        Copies the Q-network's parameters to the target network.
        """
        self._target_network.load_state_dict(self._q_network.state_dict())

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Returns Q-values predicted by the network for a single state.
        """
        assert state.shape == (2,)
        state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        pred = self._q_network(state_tensor).squeeze(0).detach().numpy()
        assert pred.shape == (self.n_actions,)
        return pred
