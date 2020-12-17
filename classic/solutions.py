import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from algo import value_iteration, monte_carlo, q_learning, policy_evaluation
from draw import draw_policy, draw_values, draw_rewards
from grid import d, w, p, r, GridWorld
from util import evaluate_policy

mpl.rcParams['figure.dpi'] = 90

GRID = [
    [d(), d(), d(), d(), d(), d()],
    [r(), w(), d(), d(), d(), d()],
    [d(), d(), d(), w(), d(), w()],
    [d(), w(), d(), d(), d(), d()],
    [d(), w(), w(), p(), w(), d()],
    [d(), d(), d(), d(), d(), d()],
]

WORLD = GridWorld(GRID, action_probability=0.75)

DISCOUNT = 0.6


def q2_1():
    policy = value_iteration(WORLD, 0.0001, DISCOUNT)
    draw_policy(WORLD, policy)

    value_function = policy_evaluation(WORLD, policy, 0.0001, DISCOUNT)

    plt.savefig(fname=f'figs/dynamic_programming_policy.pdf', dpi=90)
    plt.show()
    print('Value Function:')
    print(value_function)

    draw_values(WORLD, value_function)
    plt.savefig(fname=f'figs/dynamic_programming_value_function.pdf', dpi=90)
    plt.show()


def q2_1_2():
    """Plots episode length for varying values of gamma."""
    n_episodes = 1000
    parameter_space = np.linspace(0, 1, 200)
    lengths = np.zeros(len(parameter_space))
    for i, discount in enumerate(parameter_space):
        policy = value_iteration(WORLD, 0.0001, discount)
        lengths[i] = evaluate_policy(WORLD, policy, n_episodes)

    print(lengths)
    plt.ylabel('Average return')
    plt.xlabel(r'$\gamma$')
    plt.xlim([0, 1])
    plt.plot(parameter_space, lengths)
    plt.savefig(fname=f'figs/dynamic_programming_gamma.pdf', dpi=90)
    plt.show()


def q2_1_3():
    """Plots episode length for varying values of p."""
    n_episodes = 1000
    parameter_space = np.linspace(0, 1, 200)
    lengths = np.zeros(len(parameter_space))
    for i, prob in enumerate(parameter_space):
        world = GridWorld(GRID, action_probability=prob)
        policy = value_iteration(world, 0.0001, DISCOUNT)
        lengths[i] = evaluate_policy(world, policy, n_episodes)

    plt.ylabel('Average return')
    plt.xlabel(r'$p$')
    plt.xlim([0, 1])
    plt.plot(parameter_space, lengths)
    plt.savefig(fname=f'figs/dynamic_programming_p.pdf', dpi=90)
    plt.show()


def q2_2():
    """Plots a learning curve for Monte-Carlo."""
    n_runs = 500
    n_episodes = 300
    epsilon = 0.3
    learning_rate = 0
    t0 = time.time()

    policy, state_value_function, rewards, _ = monte_carlo(WORLD, DISCOUNT, n_episodes, epsilon, learning_rate)

    print(f'Monte-Carlo finished in {time.time() - t0}s')

    value_function = policy_evaluation(WORLD, policy, 0.0001, DISCOUNT)

    draw_values(WORLD, value_function)
    plt.savefig(fname=f'figs/monte_carlo_value_function.pdf', dpi=90)
    plt.show()

    draw_policy(WORLD, policy)
    plt.savefig(fname=f'figs/monte_carlo_policy.pdf', dpi=90)
    plt.show()

    total_rewards = np.zeros((n_runs, n_episodes))
    for i in range(n_runs):
        _, _, run_rewards, _ = monte_carlo(WORLD, DISCOUNT, n_episodes, epsilon, learning_rate)

        total_rewards[i] = run_rewards

    draw_rewards(total_rewards, color='#1f77b4')
    plt.savefig(fname=f'figs/monte_carlo_rewards.pdf', dpi=90)
    plt.show()


def q2_2_2():
    """Plots learning curves for varying values of epsilon."""
    n_runs = 500
    n_episodes = 300
    epsilons = [0, 1 / 3, 2 / 3, 1]
    epsilon_labels = {0: '0', 1 / 3: r'$\frac{1}{3}$', 2 / 3: r'$\frac{2}{3}$', 1: '1'}
    learning_rate = 0.6

    plt.xlabel('Episode number')
    plt.ylabel('Total reward')
    plt.xlim([0, 300])
    plt.ylim([-120, 50])
    plt.axhline(y=0.0, linestyle='-', color='black', linewidth=0.5)

    total_rewards = np.zeros((n_runs, n_episodes))
    for j, epsilon in enumerate(epsilons, 1):
        print(f'Running for {epsilon}')
        for i in range(n_runs):
            _, _, run_rewards, _ = monte_carlo(WORLD, DISCOUNT, n_episodes, epsilon, learning_rate)

            total_rewards[i] = run_rewards

        mean_rewards = np.mean(total_rewards, axis=0)
        plt.plot(mean_rewards, label=r'$\epsilon$ = ' + epsilon_labels[epsilon])
    plt.legend(loc='lower right')
    plt.savefig(fname=f'figs/monte_carlo_rewards_epsilon_same.pdf', dpi=90)
    plt.show()


def q2_2_3():
    """Plots learning curves for varying values of alpha."""
    n_runs = 500
    n_episodes = 300
    epsilon = 0.3
    learning_rates = [0, 1 / 3, 2 / 3, 1]
    learning_rate_labels = {0: '0', 1 / 3: r'$\frac{1}{3}$', 2 / 3: r'$\frac{2}{3}$', 1: '1'}

    plt.xlabel('Episode number')
    plt.ylabel('Total reward')
    plt.xlim([0, 300])
    plt.ylim([-120, 50])
    plt.axhline(y=0.0, linestyle='-', color='black', linewidth=0.5)

    total_rewards = np.zeros((n_runs, n_episodes))
    for j, learning_rate in enumerate(learning_rates, 5):
        print(f'Running for {learning_rate}')
        for i in range(n_runs):
            _, _, run_rewards, _ = monte_carlo(WORLD, DISCOUNT, n_episodes, epsilon, learning_rate)

            total_rewards[i] = run_rewards

        mean_rewards = np.mean(total_rewards, axis=0)
        plt.plot(mean_rewards, label=r'$\alpha$ = ' + learning_rate_labels[learning_rate])
    plt.legend(loc='lower right')
    plt.savefig(fname=f'figs/monte_carlo_rewards_alpha_same.pdf', dpi=90)
    plt.show()


def q2_3():
    """Plots a learning curve for Q-learning."""
    n_runs = 500
    n_episodes = 300
    epsilon = 0.3
    learning_rate = 0.6
    t0 = time.time()

    policy, state_value_function, rewards, _ = q_learning(WORLD, DISCOUNT, n_episodes, epsilon, learning_rate)

    print(f'Q-Learning finished in {time.time() - t0}s')

    value_function = policy_evaluation(WORLD, policy, 0.0001, DISCOUNT)
    draw_values(WORLD, value_function)
    plt.savefig(fname=f'figs/q_learning_value_function.pdf', dpi=90)
    plt.show()

    draw_policy(WORLD, policy)
    plt.savefig(fname=f'figs/q_learning_policy.pdf', dpi=90)
    plt.show()

    total_rewards = np.zeros((n_runs, n_episodes))
    for i in range(n_runs):
        _, _, run_rewards, _ = q_learning(WORLD, DISCOUNT, n_episodes, epsilon, learning_rate)

        total_rewards[i] = run_rewards

    draw_rewards(total_rewards, color='#ff7f0e')
    plt.savefig(fname=f'figs/q_learning_rewards.pdf', dpi=90)
    plt.show()


def q2_3_2():
    """Plots learning curves for varying values of epsilon."""
    n_runs = 500
    n_episodes = 300
    epsilons = [0, 1 / 3, 2 / 3, 1]
    epsilon_labels = {0: '0', 1 / 3: r'$\frac{1}{3}$', 2 / 3: r'$\frac{2}{3}$', 1: '1'}
    learning_rate = 0.6

    plt.xlabel('Episode number')
    plt.ylabel('Total reward')
    plt.xlim([0, 300])
    plt.ylim([-120, 50])
    plt.axhline(y=0.0, linestyle='-', color='black', linewidth=0.5)

    total_rewards = np.zeros((n_runs, n_episodes))
    for j, epsilon in enumerate(epsilons, 1):
        print(f'Running for {epsilon}')
        for i in range(n_runs):
            _, _, run_rewards, _ = q_learning(WORLD, DISCOUNT, n_episodes, epsilon, learning_rate)

            total_rewards[i] = run_rewards
        mean_rewards = np.mean(total_rewards, axis=0)
        plt.plot(mean_rewards, label=r'$\epsilon$ = ' + epsilon_labels[epsilon])
    plt.legend(loc='lower right')
    plt.savefig(fname=f'figs/q_learning_rewards_epsilon_same.pdf', dpi=90)
    plt.show()


def q2_3_3():
    """Plots learning curves for varying values of alpha."""
    n_runs = 500
    n_episodes = 300
    epsilon = 0.3
    learning_rates = [0, 1 / 3, 2 / 3, 1]
    learning_rate_labels = {0: '0', 1 / 3: r'$\frac{1}{3}$', 2 / 3: r'$\frac{2}{3}$', 1: '1'}

    plt.xlabel('Episode number')
    plt.ylabel('Total reward')
    plt.xlim([0, 300])
    plt.ylim([-120, 50])
    plt.axhline(y=0.0, linestyle='-', color='black', linewidth=0.5)

    total_rewards = np.zeros((n_runs, n_episodes))
    for j, learning_rate in enumerate(learning_rates, 5):
        print(f'Running for {learning_rate}')
        for i in range(n_runs):
            _, _, run_rewards, _ = q_learning(WORLD, DISCOUNT, n_episodes, epsilon, learning_rate)

            total_rewards[i] = run_rewards

        mean_rewards = np.mean(total_rewards, axis=0)
        plt.plot(mean_rewards, label=r'$\alpha$ = ' + learning_rate_labels[learning_rate])
    plt.legend(loc='lower right')
    plt.savefig(fname=f'figs/q_learning_rewards_alpha_same.pdf', dpi=90)
    plt.show()


def q2_4():
    """Plots the error curve for MC and QL."""
    n_runs = 100
    n_episodes = 100
    epsilon = 0.3
    learning_rate = 0.6

    optimal_policy = value_iteration(WORLD, 0.0001, DISCOUNT)
    optimal_V = policy_evaluation(WORLD, optimal_policy, 0.0001, DISCOUNT)

    q_errors = np.zeros((n_runs, n_episodes))
    mc_errors = np.zeros((n_runs, n_episodes))

    for i in range(n_runs):
        print(f'Running {i}')
        _, _, _, q_policies = q_learning(WORLD, DISCOUNT, n_episodes, epsilon, learning_rate)
        _, _, _, mc_policies = monte_carlo(WORLD, DISCOUNT, n_episodes, epsilon, learning_rate)
        for j in range(n_episodes):
            q_policy = q_policies[j]
            q_V = policy_evaluation(WORLD, q_policy, 0.0001, DISCOUNT)
            q_rms = np.sqrt(np.mean((q_V - optimal_V) ** 2))

            mc_policy = mc_policies[j]
            mc_V = policy_evaluation(WORLD, mc_policy, 0.0001, DISCOUNT)
            mc_rms = np.sqrt(np.mean((mc_V - optimal_V) ** 2))

            q_errors[i, j] = q_rms
            mc_errors[i, j] = mc_rms

    plt.ylabel('Optimal value function RMS')
    plt.xlabel('Number of episodes')
    plt.xlim([0, 100])
    plt.ylim([0, 12])

    mc_mean = np.mean(mc_errors, axis=0)
    plt.plot(mc_mean, color='#1f77b4', label="Monte-Carlo")

    q_mean = np.mean(q_errors, axis=0)
    plt.plot(q_mean, color='#ff7f0e', label="Q-Learning")
    plt.legend(loc="upper right")

    plt.savefig(fname=f'figs/rms.pdf', dpi=90)
    plt.show()


def q2_4_2():
    """Plots the error curve for MC and QL."""
    n_runs = 100
    n_episodes = 100

    optimal_policy = value_iteration(WORLD, 0.0001, DISCOUNT)
    optimal_V = policy_evaluation(WORLD, optimal_policy, 0.0001, DISCOUNT)

    q_errors = np.zeros((n_runs, n_episodes))
    mc_errors = np.zeros((n_runs, n_episodes))

    for i in range(n_runs):
        print(f'Running {i}')
        _, _, _, q_policies = q_learning(WORLD, DISCOUNT, n_episodes, 0.5, 0.3)
        _, _, _, mc_policies = monte_carlo(WORLD, DISCOUNT, n_episodes, 0.0, 0.5)
        for j in range(n_episodes):
            q_policy = q_policies[j]
            q_V = policy_evaluation(WORLD, q_policy, 0.0001, DISCOUNT)
            q_rms = np.sqrt(np.mean((q_V - optimal_V) ** 2))

            mc_policy = mc_policies[j]
            mc_V = policy_evaluation(WORLD, mc_policy, 0.0001, DISCOUNT)
            mc_rms = np.sqrt(np.mean((mc_V - optimal_V) ** 2))

            q_errors[i, j] = q_rms
            mc_errors[i, j] = mc_rms

    plt.ylabel('Optimal value function RMS')
    plt.xlabel('Number of episodes')
    plt.xlim([0, 100])
    plt.ylim([0, 12])

    mc_mean = np.mean(mc_errors, axis=0)
    plt.plot(mc_mean, color='#1f77b4', label="Monte-Carlo")

    q_mean = np.mean(q_errors, axis=0)
    plt.plot(q_mean, color='#ff7f0e', label="Q-Learning")
    plt.legend(loc="upper right")

    plt.savefig(fname=f'figs/rms_2.pdf', dpi=90)
    plt.show()


def q2_5_1():
    n_episodes = 50000
    epsilon = 0.3
    learning_rate = 0.6

    plt.ylabel('Value function RMS')
    plt.xlabel('Average reward')
    plt.ylim([0, 20])
    plt.xlim([-110, 20])

    optimal_policy = value_iteration(WORLD, 0.0001, DISCOUNT)
    optimal_V = policy_evaluation(WORLD, optimal_policy, 0.0001, DISCOUNT)

    x = np.zeros(n_episodes)
    y = np.zeros(n_episodes)

    _, _, mc_rewards, mc_policies = monte_carlo(WORLD, DISCOUNT, n_episodes, epsilon, learning_rate)
    for j in range(n_episodes):
        mc_policy = mc_policies[j]
        mc_V = policy_evaluation(WORLD, mc_policy, 0.0001, DISCOUNT)
        mc_rms = np.sqrt(np.mean((mc_V - optimal_V) ** 2))
        mc_reward = mc_rewards[j]

        x[j] = mc_reward
        y[j] = mc_rms

    plt.scatter(x, y, color='#1f77b4', alpha=0.05)
    poly = np.poly1d(np.polyfit(x, y, 1))
    print(poly)
    print(f'Monte-Carlo: {stats.pearsonr(x, y)}')
    plt.plot(x, poly(x), color='C2')
    plt.savefig(fname=f'figs/stupid_graph_1.pdf', dpi=90)
    plt.show()


def q2_5_2():
    n_episodes = 50000
    epsilon = 0.3
    learning_rate = 0.6

    optimal_policy = value_iteration(WORLD, 0.0001, DISCOUNT)
    optimal_V = policy_evaluation(WORLD, optimal_policy, 0.0001, DISCOUNT)

    plt.ylabel('Value function RMS')
    plt.xlabel('Average reward')
    plt.ylim([0, 20])
    plt.xlim([-110, 20])

    x = np.zeros(n_episodes)
    y = np.zeros(n_episodes)

    _, _, q_rewards, q_policies = q_learning(WORLD, DISCOUNT, n_episodes, epsilon, learning_rate)
    for j in range(n_episodes):
        q_policy = q_policies[j]
        q_V = policy_evaluation(WORLD, q_policy, 0.0001, DISCOUNT)
        q_rms = np.sqrt(np.mean((q_V - optimal_V) ** 2))
        q_reward = q_rewards[j]

        x[j] = q_reward
        y[j] = q_rms

    plt.scatter(x, y, color='#ff7f0e', alpha=0.05)
    poly = np.poly1d(np.polyfit(x, y, 1))
    print(poly)
    print(f'Q-Learning: {stats.pearsonr(x, y)}')
    plt.plot(x, poly(x), color='C2')
    plt.savefig(fname=f'figs/stupid_graph_2.pdf', dpi=90)
    plt.show()


if __name__ == "__main__":
    print('Running')
    # q2_1()
    # q2_1_2()
    # q2_1_3()
    # q2_2()
    # q2_2_2()
    q2_2_3()
    # q2_3()
    # q2_3_2()
    # q2_3_3()
    # q2_4()
    # q2_4_2()
    # q2_5_1()
    # q2_5_2()
    print("Done!")
