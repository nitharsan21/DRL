import numba

from ..do_not_touch.mdp_env_wrapper import Env1
from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
from ..to_do import Env
import numpy as np
import matplotlib.pyplot as plt


def policy_evaluation_on_line_world(theta: float = 1e-6, gamma: float = 0.99) -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """

    print('policy_evaluation_on_line_world of 7 cells :')

    # Line World of 7 cells
    nb_cells = 7
    env = Env.LineWorldMLP(7)

    # uniform random policy
    policy = np.ones([len(env.states()), len(env.actions())]) / len(env.actions())
    # policy = np.random.random((len(env.states()), len(env.actions())))

    V = np.random.random((len(env.states()),))
    V[0] = 0.0
    V[nb_cells - 1] = 0.0

    while True:
        delta = 0
        for s in env.states():
            v = V[s]
            V[s] = 0.0
            for a in env.actions():
                total = 0.0
                for s_p in env.states():
                    for r in range(len(env.rewards())):
                        total += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])
                    total *= policy[s, a]
                V[s] += total
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break
    # Env.plot_values(V,1,7)
    return V


def policy_iteration_on_line_world(theta: float = 1e-6, gamma: float = 0.99) -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """

    print('policy_iteration_on_line_world of 7 cells :')

    # Line World of 7 cells
    nb_cells = 7
    env = Env.LineWorldMLP(7)

    V = np.random.random((len(env.states()),))
    V[0] = 0.0
    V[nb_cells - 1] = 0.0

    # policy
    policy = np.random.random((len(env.states()), len(env.actions())))

    for s in env.states():
        policy[s] /= np.sum(policy[s])

    policy[0] = 0.0
    policy[nb_cells - 1] = 0.0
    # print('Initial policy : ', policy)

    while True:
        # policy evalution
        while True:
            delta = 0
            for s in env.states():
                v = V[s]
                V[s] = 0.0
                for a in env.actions():
                    total = 0.0
                    for s_p in env.states():
                        for r in range(len(env.rewards())):
                            total += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])
                        total *= policy[s, a]
                    V[s] += total
                delta = max(delta, np.abs(v - V[s]))
            if delta < theta:
                break

        # policy improvement
        stable = True
        for s in env.states():
            old_policy_s = policy[s].copy()
            q = np.zeros(len(env.actions()))
            for a in env.actions():
                for s_p in env.states():
                    for r in range(len(env.rewards())):
                        q[a] += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])
            best_a = np.argwhere(q == np.max(q)).flatten()
            policy[s] = np.sum([np.eye(len(env.actions()))[i] for i in best_a], axis=0) / len(best_a)

            if np.any(policy[s] != old_policy_s):
                stable = False
        if stable:
            # Env.plot_values(V,1,7)
            return policy, V


def value_iteration_on_line_world(theta: float = 1e-6, gamma: float = 0.99) -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """

    print('value_iteration_on_line_world of 7 cells :')

    # Line World of 7 cells
    nb_cells = 7
    env = Env.LineWorldMLP(7)

    V = np.random.random((len(env.states()),))
    V[0] = 0.0
    V[nb_cells - 1] = 0.0

    # policy
    policy = np.random.random((len(env.states()), len(env.actions())))

    for s in env.states():
        policy[s] /= np.sum(policy[s])

    policy[0] = 0.0
    policy[nb_cells - 1] = 0.0
    # print('Initial policy : ', policy)

    while True:
        delta = 0
        for s in env.states():
            v = V[s]
            q = np.zeros(len(env.actions()))
            for a in env.actions():
                total = 0.0
                for s_p in env.states():
                    for r in range(len(env.rewards())):
                        q[a] += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])
            V[s] = max(q)
            delta = max(delta, abs(V[s] - v))
        if delta < theta:
            break

    for s in env.states():
        q = np.zeros(len(env.actions()))
        for a in env.actions():
            total = 0.0
            for s_p in env.states():
                for r in range(len(env.rewards())):
                    q[a] += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])
        best_a = np.argwhere(q == np.max(q)).flatten()
        policy[s] = np.sum([np.eye(len(env.actions()))[i] for i in best_a], axis=0) / len(best_a)

    # Env.plot_values(V,1,7)
    return policy, V


def policy_evaluation_on_grid_world(theta: float = 1e-6, gamma: float = 0.99) -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    print('policy_evaluation_on_grid_world of 5x5 cells :')

    # Grid World of 5x5
    nb_cells = 5*5
    env = Env.GridWorldMLP(5,5)

    # uniform random policy
    policy = np.ones([len(env.states()), len(env.actions())]) / len(env.actions())
    # policy = np.random.random((len(env.states()), len(env.actions())))

    V = np.random.random((len(env.states()),))
    V[0] = 0.0
    V[nb_cells - 1] = 0.0

    while True:
        delta = 0
        for s in env.states():
            v = V[s]
            V[s] = 0.0
            for a in env.actions():
                total = 0.0
                for s_p in env.states():
                    for r in range(len(env.rewards())):
                        total += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])
                    total *= policy[s, a]
                V[s] += total
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break

    # Env.plot_values(V,5,5)

    return V


def policy_iteration_on_grid_world(theta: float = 1e-6, gamma: float = 0.99) -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    print('policy_iteration_on_grid_world of 5x5 cells :')

    # Grid World of 5x5
    nb_cells = 5 * 5
    env = Env.GridWorldMLP(5, 5)

    V = np.random.random((len(env.states()),))
    V[0] = 0.0
    V[nb_cells - 1] = 0.0

    # policy
    policy = np.random.random((len(env.states()), len(env.actions())))

    for s in env.states():
        policy[s] /= np.sum(policy[s])

    policy[0] = 0.0
    policy[nb_cells - 1] = 0.0
    # print('Initial policy : ', policy)

    while True:
        # policy evalution
        while True:
            delta = 0
            for s in env.states():
                v = V[s]
                V[s] = 0.0
                for a in env.actions():
                    total = 0.0
                    for s_p in env.states():
                        for r in range(len(env.rewards())):
                            total += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])
                        total *= policy[s, a]
                    V[s] += total
                delta = max(delta, np.abs(v - V[s]))
            if delta < theta:
                break

        # policy improvement
        stable = True
        for s in env.states():
            old_policy_s = policy[s].copy()
            q = np.zeros(len(env.actions()))
            for a in env.actions():
                for s_p in env.states():
                    for r in range(len(env.rewards())):
                        q[a] += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])
            best_a = np.argwhere(q == np.max(q)).flatten()
            policy[s] = np.sum([np.eye(len(env.actions()))[i] for i in best_a], axis=0) / len(best_a)

            if np.any(policy[s] != old_policy_s):
                stable = False
        if stable:
            # Env.plot_values(V,5,5)
            return policy, V


def value_iteration_on_grid_world(theta: float = 1e-6, gamma: float = 0.99) -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """

    print('value_iteration_on_grid_world of 5x5 cells :')

    # Grid World of 5x5
    nb_cells = 5 * 5
    env = Env.GridWorldMLP(5, 5)

    V = np.random.random((len(env.states()),))
    V[0] = 0.0
    V[nb_cells - 1] = 0.0

    # policy
    policy = np.random.random((len(env.states()), len(env.actions())))

    for s in env.states():
        policy[s] /= np.sum(policy[s])

    policy[0] = 0.0
    policy[nb_cells - 1] = 0.0
    # print('Initial policy : ', policy)

    while True:
        delta = 0
        for s in env.states():
            v = V[s]
            q = np.zeros(len(env.actions()))
            for a in env.actions():
                total = 0.0
                for s_p in env.states():
                    for r in range(len(env.rewards())):
                        q[a] += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])
            V[s] = max(q)
            delta = max(delta, abs(V[s] - v))
        if delta < theta:
            break

    for s in env.states():
        q = np.zeros(len(env.actions()))
        for a in env.actions():
            total = 0.0
            for s_p in env.states():
                for r in range(len(env.rewards())):
                    q[a] += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])
        best_a = np.argwhere(q == np.max(q)).flatten()
        policy[s] = np.sum([np.eye(len(env.actions()))[i] for i in best_a], axis=0) / len(best_a)
    # Env.plot_values(V,5,5)
    return policy, V


def policy_evaluation_on_secret_env1(theta: float = 1e-6, gamma: float = 0.99) -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    print('policy_evaluation_on_secret_env1 : ')

    # Creates a Secret Env1
    env = Env1()

    # uniform random policy
    policy = np.ones([len(env.states()), len(env.actions())]) / len(env.actions())
    # policy = np.random.random((len(env.states()), len(env.actions())))

    V = np.random.random((len(env.states()),))
    V[0] = 0.0
    V[len(env.states()) - 1] = 0.0

    p = np.zeros((len(env.states()), len(env.actions()), len(env.states()), len(env.rewards())))

    while True:
        delta = 0
        for s in env.states():
            v = V[s]
            V[s] = 0.0
            for a in env.actions():
                total = 0.0
                for s_p in env.states():
                    for r in range(len(env.rewards())):
                        total += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])
                    total *= policy[s, a]
                V[s] += total
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break
    return V


def policy_iteration_on_secret_env1(theta: float = 1e-6, gamma: float = 0.99) -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    print('policy evaluation on a Secret Env1 :')

    # Line World of 7 cells
    env = Env1()
    print(env.states())

    V = np.random.random((len(env.states()),))
    V[0] = 0.0
    V[len(env.states()) - 1] = 0.0

    # policy
    policy = np.random.random((len(env.states()), len(env.actions())))

    for s in env.states():
        policy[s] /= np.sum(policy[s])

    policy[0] = 0.0
    policy[len(env.states()) - 1] = 0.0
    # print('Initial policy : ', policy)

    while True:
        # policy evalution
        while True:
            delta = 0
            for s in env.states():
                v = V[s]
                V[s] = 0.0
                for a in env.actions():
                    total = 0.0
                    for s_p in env.states():
                        for r in range(len(env.rewards())):
                            total += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])
                        total *= policy[s, a]
                    V[s] += total
                delta = max(delta, np.abs(v - V[s]))
            if delta < theta:
                break

        # policy improvement
        stable = True
        for s in env.states():
            old_policy_s = policy[s].copy()
            q = np.zeros(len(env.actions()))
            for a in env.actions():
                total = 0.0
                for s_p in env.states():
                    for r in range(len(env.rewards())):
                        q[a] += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])
            best_a = np.argwhere(q == np.max(q)).flatten()
            policy[s] = np.sum([np.eye(len(env.actions()))[i] for i in best_a], axis=0) / len(best_a)

            if np.any(policy[s] != old_policy_s):
                stable = False
        if stable:
            return policy, V


def value_iteration_on_secret_env1(theta: float = 1e-6, gamma: float = 0.99) -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """

    print('value_iteration_on_secret_env1 :')

    env = Env1()

    V = np.random.random((len(env.states()),))
    V[0] = 0.0
    V[len(env.states()) - 1] = 0.0

    # policy
    policy = np.random.random((len(env.states()), len(env.actions())))

    for s in env.states():
        policy[s] /= np.sum(policy[s])

    policy[0] = 0.0
    policy[len(env.states()) - 1] = 0.0
    # print('Initial policy : ', policy)

    while True:
        delta = 0
        for s in env.states():
            v = V[s]
            q = np.zeros(len(env.actions()))
            for a in env.actions():
                total = 0.0
                for s_p in env.states():
                    for r in range(len(env.rewards())):
                        q[a] += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])

            V[s] = max(q)
            delta = max(delta, abs(V[s] - v))
        if delta < theta:
            break

    for s in env.states():
        q = np.zeros(len(env.actions()))
        for a in env.actions():
            for s_p in env.states():
                for r in range(len(env.rewards())):
                    q[a] += env.transition_probability(s, a, s_p, r) * (env.rewards()[r] + gamma * V[s_p])
        best_a = np.argwhere(q == np.max(q)).flatten()
        policy[s] = np.sum([np.eye(len(env.actions()))[i] for i in best_a], axis=0) / len(best_a)
    return policy, V


def demo():
    print(policy_evaluation_on_line_world())
    # print(policy_iteration_on_line_world())
    # print(value_iteration_on_line_world())
    # #
    # print(policy_evaluation_on_grid_world())
    # print(policy_iteration_on_grid_world())
    # print(value_iteration_on_grid_world())
    #
    # print(policy_evaluation_on_secret_env1())
    # print(policy_iteration_on_secret_env1())
    # print(value_iteration_on_secret_env1())
