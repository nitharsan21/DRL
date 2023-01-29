from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env3
import random
import numpy as np
from ..to_do import Env


def sarsa_on_tic_tac_toe_solo( max_iter_count: int = 10000,
               gamma: float = 0.99,
               alpha: float = 0.05,
               epsilon: float = 0.5) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """

    env = Env.TicTacToeEnv()
    q = {}
    for it in range(max_iter_count):

        if env.is_game_over():
            env.reset()

        s = env.state_id()
        aa = env.available_actions_ids()
        if s not in q:
            q[s] = {}
            for a in aa:
                q[s][a] = 0.0 if env.is_game_over() else random.random()

        if random.random() <= epsilon:
            a = np.random.choice(aa)
        else:
            a = aa[np.argmax([q[s][a] for a in aa])]

        old_score = env.score()
        env.act_with_action_id(a)
        new_score = env.score()
        r = new_score - old_score

        if env.is_game_over():
            env.reset()
        s_p = env.state_id()
        aa_p = env.available_actions_ids()

        if s_p not in q:
            q[s_p] = {}
            for a in aa_p:
                q[s_p][a] = 0.0 if env.is_game_over() else random.random()

        predict = q[s][a]
        target = 0.0
        for i in aa_p:
            target = r + gamma * q[s_p][i]
        q[s][a] += alpha * (target - predict)


    pi = {}
    for (s, a_dict) in q.items():
        pi[s] = {}
        actions = []
        q_values = []
        for (a, q_value) in a_dict.items():
            actions.append(a)
            q_values.append(q_value)

        best_action_idx = np.argmax(q_values)
        for i in range(len(actions)):
            pi[s][actions[i]] = 1.0 if actions[i] == best_action_idx else 0.0

    return q, pi


def q_learning_on_tic_tac_toe_solo( max_iter_count: int = 10000,
               gamma: float = 0.99,
               alpha: float = 0.1,
               epsilon: float = 0.2) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    q = {}
    env = Env.TicTacToeEnv()

    for it in range(max_iter_count):

        if env.is_game_over():
            env.reset()

        s = env.state_id()
        aa = env.available_actions_ids()
        if s not in q:
            q[s] = {}
            for a in aa:
                q[s][a] = 0.0 if env.is_game_over() else random.random()

        if random.random() <= epsilon:
            a = np.random.choice(aa)
        else:
            a = aa[np.argmax([q[s][a] for a in aa])]

        old_score = env.score()
        env.act_with_action_id(a)
        new_score = env.score()
        r = new_score - old_score

        if env.is_game_over():
            env.reset()

        s_p = env.state_id()
        aa_p = env.available_actions_ids()

        if s_p not in q:
            q[s_p] = {}
            for a in aa_p:
                q[s_p][a] = 0.0 if env.is_game_over() else random.random()

        q[s][a] += alpha * (r + gamma * np.max([q[s_p][a] for a in aa_p]) - q[s][a])

    pi = {}
    for (s, a_dict) in q.items():
        pi[s] = {}
        actions = []
        q_values = []
        for (a, q_value) in a_dict.items():
            actions.append(a)
            q_values.append(q_value)

        best_action_idx = np.argmax(q_values)
        for i in range(len(actions)):
            pi[s][actions[i]] = 1.0 if actions[i] == best_action_idx else 0.0

    return q, pi




def expected_sarsa_on_tic_tac_toe_solo(max_iter_count: int = 10000,
               gamma: float = 0.99,
               alpha: float = 0.1,
               epsilon: float = 0.2) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env.TicTacToeEnv()
    q = {}

    for it in range(max_iter_count):

        if env.is_game_over():
            env.reset()

        s = env.state_id()
        aa = env.available_actions_ids()
        if s not in q:
            q[s] = {}
            for a in aa:
                q[s][a] = 0.0 if env.is_game_over() else random.random()

        if random.random() <= epsilon:
            a = np.random.choice(aa)
        else:
            a = aa[np.argmax([q[s][a] for a in aa])]

        old_score = env.score()
        env.act_with_action_id(a)
        new_score = env.score()
        r = new_score - old_score

        if env.is_game_over():
            env.reset()

        s_p = env.state_id()
        aa_p = env.available_actions_ids()

        if s_p not in q:
            q[s_p] = {}
            for a in aa_p:
                q[s_p][a] = 0.0 if env.is_game_over() else random.random()

        expected_q = 0
        q_max = np.max([q[s_p][a] for a in aa_p])
        greedy_actions = 0

        for i in aa_p:
            if q[s_p][i] == q_max:
                greedy_actions += 1

        non_greedy_action_probability = epsilon / len(aa_p)
        greedy_action_probability = ((1 - epsilon) / greedy_actions) + non_greedy_action_probability

        for i in aa_p:
            if q[s_p][i] == q_max:
                expected_q += q[s_p][i] * greedy_action_probability
            else:
                expected_q += q[s_p][i] * non_greedy_action_probability

        q[s][a] += alpha * (r + gamma * expected_q - q[s][a])

    pi = {}
    for (s, a_dict) in q.items():
        pi[s] = {}
        actions = []
        q_values = []
        for (a, q_value) in a_dict.items():
            actions.append(a)
            q_values.append(q_value)

        best_action_idx = np.argmax(q_values)
        for i in range(len(actions)):
            pi[s][actions[i]] = 1.0 if actions[i] == best_action_idx else 0.0

    return q, pi


def sarsa_on_secret_env3( max_iter_count: int = 10000,
               gamma: float = 0.99,
               alpha: float = 0.1,
               epsilon: float = 0.2) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    q = {}
    for it in range(max_iter_count):

        if env.is_game_over():
            env.reset()

        s = env.state_id()
        aa = env.available_actions_ids()
        if s not in q:
            q[s] = {}
            for a in aa:
                q[s][a] = 0.0 if env.is_game_over() else random.random()

        if random.random() <= epsilon:
            a = np.random.choice(aa)
        else:
            a = aa[np.argmax([q[s][a] for a in aa])]

        old_score = env.score()
        env.act_with_action_id(a)
        new_score = env.score()
        r = new_score - old_score

        s_p = env.state_id()
        aa_p = env.available_actions_ids()

        if s_p not in q:
            q[s_p] = {}
            for a in aa_p:
                q[s_p][a] = 0.0 if env.is_game_over() else random.random()

        predict = q[s][a]
        target = 0.0
        for i in aa_p:
            target = r + gamma * q[s_p][i]
        q[s][a] += alpha * (target - predict)

    pi = {}
    for (s, a_dict) in q.items():
        pi[s] = {}
        actions = []
        q_values = []
        for (a, q_value) in a_dict.items():
            actions.append(a)
            q_values.append(q_value)

        best_action_idx = np.argmax(q_values)
        for i in range(len(actions)):
            pi[s][actions[i]] = 1.0 if actions[i] == best_action_idx else 0.0

    return q, pi



def q_learning_on_secret_env3( max_iter_count: int = 10000,
               gamma: float = 0.99,
               alpha: float = 0.1,
               epsilon: float = 0.2) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    q = {}

    for it in range(max_iter_count):

        if env.is_game_over():
            env.reset()

        s = env.state_id()
        aa = env.available_actions_ids()
        if s not in q:
            q[s] = {}
            for a in aa:
                q[s][a] = 0.0 if env.is_game_over() else random.random()

        if random.random() <= epsilon:
            a = np.random.choice(aa)
        else:
            a = aa[np.argmax([q[s][a] for a in aa])]

        old_score = env.score()
        env.act_with_action_id(a)
        new_score = env.score()
        r = new_score - old_score

        if env.is_game_over():
            env.reset()

        s_p = env.state_id()
        aa_p = env.available_actions_ids()

        if s_p not in q:
            q[s_p] = {}
            for a in aa_p:
                q[s_p][a] = 0.0 if env.is_game_over() else random.random()

        q[s][a] += alpha * (r + gamma * np.max([q[s_p][a] for a in aa_p]) - q[s][a])

    pi = {}
    for (s, a_dict) in q.items():
        pi[s] = {}
        actions = []
        q_values = []
        for (a, q_value) in a_dict.items():
            actions.append(a)
            q_values.append(q_value)

        best_action_idx = np.argmax(q_values)
        for i in range(len(actions)):
            pi[s][actions[i]] = 1.0 if actions[i] == best_action_idx else 0.0

    return q, pi



def expected_sarsa_on_secret_env3( max_iter_count: int = 10000,
               gamma: float = 0.99,
               alpha: float = 0.1,
               epsilon: float = 0.2) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    q = {}

    for it in range(max_iter_count):

        if env.is_game_over():
            env.reset()

        s = env.state_id()
        aa = env.available_actions_ids()
        if s not in q:
            q[s] = {}
            for a in aa:
                q[s][a] = 0.0 if env.is_game_over() else random.random()

        if random.random() <= epsilon:
            a = np.random.choice(aa)
        else:
            a = aa[np.argmax([q[s][a] for a in aa])]

        old_score = env.score()
        env.act_with_action_id(a)
        new_score = env.score()
        r = new_score - old_score

        if env.is_game_over():
            env.reset()

        s_p = env.state_id()
        aa_p = env.available_actions_ids()

        if s_p not in q:
            q[s_p] = {}
            for a in aa_p:
                q[s_p][a] = 0.0 if env.is_game_over() else random.random()

        expected_q = 0
        q_max = np.max([q[s_p][a] for a in aa_p])
        greedy_actions = 0

        for i in aa_p:
            if q[s_p][i] == q_max:
                greedy_actions += 1

        non_greedy_action_probability = epsilon / len(aa_p)
        greedy_action_probability = ((1 - epsilon) / greedy_actions) + non_greedy_action_probability

        for i in aa_p:
            if q[s_p][i] == q_max:
                expected_q += q[s_p][i] * greedy_action_probability
            else:
                expected_q += q[s_p][i] * non_greedy_action_probability

        q[s][a] += alpha * (r + gamma * expected_q - q[s][a])

    pi = {}
    for (s, a_dict) in q.items():
        pi[s] = {}
        actions = []
        q_values = []
        for (a, q_value) in a_dict.items():
            actions.append(a)
            q_values.append(q_value)

        best_action_idx = np.argmax(q_values)
        for i in range(len(actions)):
            pi[s][actions[i]] = 1.0 if actions[i] == best_action_idx else 0.0

    return q, pi

def demo():
    print(sarsa_on_tic_tac_toe_solo())
    # print(q_learning_on_tic_tac_toe_solo())
    # print(expected_sarsa_on_tic_tac_toe_solo())
    #
    # print(sarsa_on_secret_env3())
    # print(q_learning_on_secret_env3())
    # print(expected_sarsa_on_secret_env3())
