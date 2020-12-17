import matplotlib.pyplot as plt
import numpy as np

from agent import Agent


def plot_loss(agent: Agent, suffix: str = ''):
    fig, ax = plt.subplots()
    plt.yscale('log')
    ax.set_title("Loss against Episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    losses = np.convolve(np.asarray(agent.losses), np.ones(1000) / 1000, mode='valid')
    ax.plot(losses)
    fig.savefig(f'plots/losses{suffix}.png')
    plt.close(fig)


def plot_distances(agent: Agent, suffix: float = ''):
    fig, ax = plt.subplots()
    ax.set_title("Distance against Episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Distance")
    ax.plot(agent.distances)
    fig.savefig(f'plots/distances{suffix}.png')
    plt.close(fig)


def plot_reward(agent: Agent, suffix: str = ''):
    fig, ax = plt.subplots()
    ax.set_title("Average Reward against Episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.plot(agent.rewards)
    fig.savefig(f'plots/rewards{suffix}.png')
    plt.close(fig)


def plot_trace(trace, env, suffix: str = ''):
    for i in range(len(trace) - 1):
        env.draw_line(trace[i], trace[i + 1])


def visualise(agent: Agent, suffix: str = ''):
    fig, ax = plt.subplots()
    ax.set_title(hyperparameters(agent))

    arrows = [r'$\rightarrow$', r'$\uparrow$', r'$\downarrow$', r'$\nearrow$', r'$\searrow$']
    dqn = agent.dqn
    DIM = 50
    xs = np.linspace(0, 1, DIM)
    ys = np.linspace(0, 1, DIM)
    q_values = np.zeros((DIM, DIM))
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            state = np.array([x, y])
            qs = dqn.predict(state)
            q_values[row][col] = qs.mean()
            arrow = arrows[np.argmax(qs).item()]
            plt.text(col - 0.5, row - 0.5, arrow, size=6)

    h = ax.imshow(q_values, origin='lower')
    fig.colorbar(h, ax=ax)
    fig.savefig(f'plots/q_values{suffix}.png')
    plt.close(fig)
    # plt.show()


def hyperparameters(agent: Agent):
    return fr'N = {agent.batch_size}, $\gamma$ = {agent.discount}, $\alpha$ = {agent.learning_rate}, $\varepsilon$ = {agent.episode_epsilon}, Episode length = {agent.episode_length}'
