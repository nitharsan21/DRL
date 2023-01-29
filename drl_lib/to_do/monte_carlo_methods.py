from collections import defaultdict

from random import random, choice, choices

import numpy as np

from ..to_do import Env


from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env2



def monte_carlo_es_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """

    max_episodes = 500
    env = Env.TicTacToeEnv()
    actions = env.available_actions_ids()
    pi = defaultdict(lambda: {a: random() for a in actions})
    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    Returns = defaultdict()

    for ep in range(max_episodes):
        env.reset()
        s0 = env.state_id()
        pis = [pi[s0][a] for a in env.available_actions_ids()]
        a0 = choices(env.available_actions_ids(), weights=pis)[0]
        s = s0
        a = a0

        env.act_with_action_id(env.currentplayer.play(env.available_actions_ids(), env.state_id(), pis))

        s_history = [s0]
        a_history = [a0]
        s_p_history = [env.state_id()]
        r_history = [env.score()]

        while (not env.is_game_over() or env.available_actions_ids() != []):
            s = env.state_id()
            pis = [pi[s][a] for a in env.available_actions_ids()]

            if (not env.is_game_over()):
                a = choices(env.available_actions_ids(), weights=pis)[0]
                # faire jouer player[1]
                env.act_with_action_id(env.currentplayer.play(env.available_actions_ids(), env.state_id(), pis))

            s_history.append(s)
            a_history.append(a)
            s_p_history.append(env.state_id())
            r_history.append(env.score())

        G = 0
        for t in reversed(range(len(s_history))):
            G = 0.999 * G + r_history[t]
            s_t = s_history[t]
            a_t = a_history[t]

            appear = False
            for t_p in range(t - 1):
                if s_history[t_p] == s_t and a_history[t_p] == a_t:
                    appear = True
                    break
            if appear:
                continue

            Returns[s_t][a_t].append(G)
            Q[s_t, a_t] = np.mean(Returns[s_t][a_t])
            pi[s_t, :] = 0.0
            pi[s_t, np.argmax(Q[s_t])] = 1.0

    return pi, Q


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    pass


def off_policy_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env.TicTacToeEnv()

    actions = env.available_actions_ids()

    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    C = defaultdict(lambda: {a: 0.0 for a in actions})

    pi = defaultdict(lambda: {a: random() for a in actions})
    target_policy = pi
    num_episodes = 20

    for i_episode in range(1, num_episodes + 1):
        env.reset()
        s0 = env.state_id()
        pis = [pi[s0][a] for a in env.available_actions_ids()]
        a0 = choices(env.available_actions_ids(), weights=pis)[0]

        # faire jouer player[1]
        env.act_with_action_id(env.currentplayer.play(env.available_actions_ids(), env.state_id(), pis))

        s_history = [s0]
        a_history = [a0]
        s_p_history = [env.state_id()]
        r_history = [env.score()]

        while (not env.is_game_over() or env.available_actions_ids() != []):
            s = env.state_id()
            pis = [pi[s][a] for a in env.available_actions_ids()]

            if (not env.is_game_over()):
                a = choices(env.available_actions_ids(), weights=pis)[0]
                # faire jouer player[1]
                env.act_with_action_id(env.currentplayer.play(env.available_actions_ids(), env.state_id(), pis))

            s_history.append(s)
            a_history.append(a)
            s_p_history.append(env.state_id())
            r_history.append(env.score())

        G = 0.0
        W = 1.0
        delta = 0.999

        for t in range(len(s_p_history))[::-1]:
            state, action, reward = s_p_history[t], a_history[t], r_history[t]
            G = delta * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            target_policy[state] = {a: 0.0 for a in actions}
            best_action = max(Q[state], key=Q[state].get)
            target_policy[state][best_action] = 1.0

            if action != best_action:
                break

            W = W * (target_policy[state][action] / pi[state][action])

    return Q, target_policy




def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """

    max_episodes = 500
    env = Env2()
    actions = env.available_actions_ids()
    pi = defaultdict(lambda: {a: random() for a in actions})
    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    Returns = defaultdict()

    for ep in range(max_episodes):
        env.reset()
        s0 = env.state_id()
        pis = [pi[s0][a] for a in env.available_actions_ids()]
        a0 = choices(env.available_actions_ids(), weights=pis)[0]
        s = s0
        a = a0

        env.act_with_action_id(a0)

        s_history = [s0]
        a_history = [a0]
        s_p_history = [env.state_id()]
        r_history = [env.score()]

        while (not env.is_game_over() or env.available_actions_ids() != []):
            s = env.state_id()
            pis = [pi[s][a] for a in env.available_actions_ids()]

            if (not env.is_game_over()):
                a = choices(env.available_actions_ids(), weights=pis)[0]
                # faire jouer player[1]
                env.act_with_action_id(a)

            s_history.append(s)
            a_history.append(a)
            s_p_history.append(env.state_id())
            r_history.append(env.score())

        G = 0
        for t in reversed(range(len(s_history))):
            G = 0.999 * G + r_history[t]
            s_t = s_history[t]
            a_t = a_history[t]

            appear = False
            for t_p in range(t - 1):
                if s_history[t_p] == s_t and a_history[t_p] == a_t:
                    appear = True
                    break
            if appear:
                continue

            Returns[s_t][a_t].append(G)
            Q[s_t, a_t] = np.mean(Returns[s_t][a_t])
            pi[s_t, :] = 0.0
            pi[s_t, np.argmax(Q[s_t])] = 1.0

    return pi, Q
    pass


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """

    pass


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    actions = env.available_actions_ids()

    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    C = defaultdict(lambda: {a: 0.0 for a in actions})

    pi = defaultdict(lambda: {a: random() for a in actions})
    target_policy = pi
    num_episodes = 20

    for i_episode in range(1, num_episodes + 1):
        env.reset()
        s0 = env.state_id()
        pis = [pi[s0][a] for a in env.available_actions_ids()]
        a0 = choices(env.available_actions_ids(), weights=pis)[0]

        # faire jouer player[1]
        env.act_with_action_id(a0)

        s_history = [s0]
        a_history = [a0]
        s_p_history = [env.state_id()]
        r_history = [env.score()]

        while (not env.is_game_over() or env.available_actions_ids() != []):
            s = env.state_id()
            pis = [pi[s][a] for a in env.available_actions_ids()]

            if (not env.is_game_over()):
                a = choices(env.available_actions_ids(), weights=pis)[0]
                # faire jouer player[1]
                env.act_with_action_id(a)

            s_history.append(s)
            a_history.append(a)
            s_p_history.append(env.state_id())
            r_history.append(env.score())

        G = 0.0
        W = 1.0
        delta = 0.999

        for t in range(len(s_p_history))[::-1]:
            state, action, reward = s_p_history[t], a_history[t], r_history[t]
            G = delta * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            target_policy[state] = {a: 0.0 for a in actions}
            best_action = max(Q[state], key=Q[state].get)
            target_policy[state][best_action] = 1.0

            if action != best_action:
                break

            W = W * (target_policy[state][action] / pi[state][action])

    return Q, target_policy



def demo():
    # print(monte_carlo_es_on_tic_tac_toe_solo())
    # print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    # print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

    # print(monte_carlo_es_on_secret_env2())
    # print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    print(off_policy_monte_carlo_control_on_secret_env2())
