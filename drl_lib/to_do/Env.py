import random

import numpy as np
import numba
import numpy.random
from more_itertools import take
from tqdm import tqdm
from typing import Optional
import matplotlib.pyplot as plt


class SingleAgentEnv:
    def state_id(self) -> int:
        pass

    def is_game_over(self) -> bool:
        pass

    def act_with_action_id(self, action_id: int):
        pass

    def score(self) -> float:
        pass

    def available_actions_ids(self) -> np.ndarray:
        pass

    def reset(self):
        pass

    def view(self):
        pass

    def reset_random(self):
        pass


class MDPEnv:
    def states(self) -> np.ndarray:
        pass

    def actions(self) -> np.ndarray:
        pass

    def rewards(self) -> np.ndarray:
        pass

    def is_state_terminal(self, s: int) -> bool:
        pass

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        pass

    def view_state(self, s: int):
        pass


"""
        Enviroment Line World
"""


class LineWorldMLP(MDPEnv):
    def __init__(self, nb_cells: int = 5):
        self.nb_cells = nb_cells
        self.state = np.arange(nb_cells)
        self.action = np.array([0, 1])
        self.reward = np.array([-1, 0, 1])

        self.probability = np.zeros((len(self.state), len(self.action), len(self.state), len(self.reward)))

        for s in self.state[1:-1]:
            if s == 1:
                self.probability[s, 0, s - 1, 0] = 1.0
            else:
                self.probability[s, 0, s - 1, 1] = 1.0

            if s == nb_cells - 2:
                self.probability[s, 1, s + 1, 2] = 1.0
            else:
                self.probability[s, 1, s + 1, 1] = 1.0

    def states(self) -> np.ndarray:
        return self.state

    def actions(self) -> np.ndarray:
        return self.action

    def rewards(self) -> np.ndarray:
        return self.reward

    def is_state_terminal(self, s: int) -> bool:
        return s == 0 or s == self.nb_cells - 1

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self.probability[s, a, s_p, r]

    def view_state(self, s: int):
        for i in self.state:
            if i == s:
                print("X", end='')
            else:
                print("_", end='')
        print()


class GridWorldMLP(MDPEnv):
    def __init__(self, rows: int = 5, cols: int = 5):
        self.rows = rows
        self.cols = cols
        self.state = np.arange(rows * cols)
        self.nbcells = rows * cols
        self.action = np.array([0, 1, 2, 3])
        self.reward = np.array([-1.0, 0.0, 1.0])

        self.probability = np.zeros((len(self.state), len(self.action), len(self.state), len(self.reward)))

        def to_s(row, col):
            return ((row * self.cols) + (col + 1)) - 1

        def inc(row, col, a):
            if a == 0:  # left
                col = max(col - 1, 0)
            elif a == 1:  # right
                col = min(col + 1, self.cols - 1)
            elif a == 2:  # up
                row = max(row - 1, 0)
            elif a == 3:  # down
                row = min(row + 1, self.rows - 1)
            return (row, col)

        for row in range(rows):
            for col in range(cols):
                s = to_s(row, col)
                if s in self.state[1:-1]:
                    for a in self.action:
                        r, c = inc(row, col, a)
                        s_p = to_s(r, c)
                        if s != s_p:
                            if (s == 1 and a == 0) or (s == to_s(1, 0) and a == 2):
                                self.probability[s, a, s_p, 0] = 1.0
                            elif (s == self.nbcells - 2 and a == 1) or (s == to_s(rows - 2, cols - 1) and a == 3):
                                self.probability[s, a, s_p, 2] = 1.0
                            else:
                                self.probability[s, a, s_p, 1] = 1.0

    def states(self) -> np.ndarray:
        return self.state

    def actions(self) -> np.ndarray:
        return self.action

    def rewards(self) -> np.ndarray:
        return self.reward

    def is_state_terminal(self, s: int) -> bool:
        return s == 0 or s == self.nb_cells - 1

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self.probability[s, a, s_p, r]

    def viewstate(self, s: int):
        for i in range(self.rows):
            for j in range(self.cols):
                if s == (((i * self.cols) + (j + 1)) - 1):
                    print("X", end='')
                else:
                    print("", end='')
            print('\n')
        print()


def plot_values(V, row, colonne):
    # reshape value function
    V_sq = np.reshape(V, (row, colonne))

    # plot the state-value function
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    im = ax.imshow(V_sq, cmap='cool')
    for (j, i), label in np.ndenumerate(V_sq):
        ax.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=14)
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.title('State-Value Function')
    plt.show()


"""
    Enviroment Single Agent

"""


class Player:
    def __init__(self, sign: int, type: str) -> None:
        self.sign = sign
        self.is_winner = False
        self.type = type  # 'H':human player, 'R': Random player, 'A'

    def play(self, available_actions, state_id=None, policy=None) -> Optional[int]:
        action_id = None
        if self.type == 'H':
            while action_id not in available_actions:
                print(available_actions)
                action_id = int(input("Please enter your action id IN : "))
        elif self.type == 'R':
            action_id = np.random.choice(available_actions)
        else:
            if state_id is not None and policy is not None:
                # action_id = max(policy[state_id], key=policy[state_id].get)
                action_id = random.choices(available_actions, weights=policy)[0]
                if (action_id == None):
                    action_id = np.random.choice(available_actions)
        return action_id


class TicTacToeEnv(SingleAgentEnv):
    def __init__(self, size=3) -> None:
        self.size = size
        self.board = np.zeros((size, size))
        self.actions = np.arange(size * size)
        self.players = np.array([Player(1, 'R'), Player(2, 'A')])
        self.currentplayer = self.players[0]

    def state_id(self) -> int:
        state = 0
        for i in range(self.size):
            for j in range(self.size):
                state += self.board[i][j] * pow(self.size, i * self.size + j)
        return int(state)

    def is_game_over(self) -> bool:
        if len(self.available_actions_ids()) == 0:
            return True
        else:
            # Check diagonals
            if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != 0:
                if self.players[0].sign == self.board[0][0]:
                    self.players[0].is_winner = True
                else:
                    self.players[1].is_winner = True
                return True
            elif self.board[2][0] == self.board[1][1] == self.board[0][2] and self.board[2][0] != 0:
                if self.players[0].sign == self.board[2][0]:
                    self.players[0].is_winner = True
                else:
                    self.players[1].is_winner = True
                return True
            else:
                for i in range(self.size):
                    # Check horizontals
                    if self.board[i][0] == self.board[i][1] == self.board[i][2] and self.board[i][0] != 0:
                        if self.players[0].sign == self.board[i][0]:
                            self.players[0].is_winner = True
                        else:
                            self.players[1].is_winner = True
                        return True
                    # Check verticals
                    elif self.board[0][i] == self.board[1][i] == self.board[2][i] and self.board[0][i] != 0:
                        if self.players[0].sign == self.board[0][0]:
                            self.players[0].is_winner = True
                        else:
                            self.players[1].is_winner = True
                        return True
        return False

    def act_with_action_id(self, action_id: int):
        print(action_id)
        i = action_id // self.size
        j = action_id % self.size
        self.board[i][j] = self.currentplayer.sign

        if (self.currentplayer == self.players[0]):
            self.currentplayer = self.players[1]
        else:
            self.currentplayer = self.players[0]

        print(self.convertStateToBoard(self.state_id()))

    def score(self) -> float:
        score = 0
        if (self.is_game_over()):
            if self.players[1].is_winner:
                score = 10
            elif self.players[0].is_winner == False and self.players[1].is_winner == False:
                score = 0
            else:
                score = -1

            self.reset()
        return 0

    def available_actions_ids(self) -> np.ndarray:
        positions = []
        cpt = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    positions.append(cpt)
                cpt += 1

        return np.array(positions)

    def reset(self):
        self.board = np.zeros((self.size, self.size))
        self.players[0].is_winner = False
        self.players[1].is_winner = False
        self.currentplayer = self.players[0]

    def convertStateToBoard(self, state, b=3):
        if state == 0:
            return np.zeros((self.size, self.size))
        digits = []
        while len(digits) < self.size * self.size:
            digits.append(int(state % b))
            state //= b
        digits = np.array(digits)
        return digits.reshape(self.size, self.size)

    def playWith(self, action):
        self.act_with_action_id(self.currentplayer.sign, action)
        if (self.currentplayer == self.players[0]):
            self.currentplayer = self.players[1]
        else:
            self.currentplayer = self.players[0]

        return self.state_id, self.score(), self.is_game_over()
