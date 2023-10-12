import sys
from typing import Callable, Tuple, Union
from sys import maxsize as inf

from adversarialsearchproblem import (
    Action,
    AdversarialSearchProblem,
    State as GameState,
)


def minimax(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    """
    Implement the minimax algorithm on ASPs, assuming that the given game is
    both 2-player and constant-sum.

    Input:
        asp - an AdversarialSearchProblem
    Output:
        an action (an element of asp.get_available_actions(asp.get_start_state()))
    """
    player = asp._start_state.player_to_move()
    value, move = max_value(asp, asp.get_start_state(), player)
    return move

def max_value(asp: AdversarialSearchProblem[GameState, Action], state, player):
    if asp.is_terminal_state(state):
        return asp.evaluate_terminal(state)[player], None
    v = -sys.maxsize**1000
    move = None
    for a in asp.get_available_actions(state):
        v2, a2 = min_value(asp, asp.transition(state, a), player)
        if v2>v:
            v, move = v2, a
    return v,move

def min_value(asp: AdversarialSearchProblem[GameState, Action], state, player):
    if asp.is_terminal_state(state):
        return asp.evaluate_terminal(state)[player], None
    v = sys.maxsize**1000
    move = None
    for a in asp.get_available_actions(state):
        v2,a2 = max_value(asp, asp.transition(state, a), player)
        if v2<v:
            v, move = v2, a
    return v, move


def alpha_beta(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    """
    Implement the alpha-beta pruning algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum.

    Input:
        asp - an AdversarialSearchProblem
    Output:
        an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    player = asp._start_state.player_to_move()
    value, move = max_valueab(asp, asp.get_start_state(), -sys.maxsize**1000, sys.maxsize**1000, player)
    return move

def max_valueab(asp: AdversarialSearchProblem[GameState, Action], state, a, b, player):
    if asp.is_terminal_state(state):
        return asp.evaluate_terminal(state)[player], None
    v = -sys.maxsize**1000
    move = None
    for ac in asp.get_available_actions(state):
        v2, ac2 = min_valueab(asp, asp.transition(state, ac), a, b, player)
        if v2>v:
            v, move = v2, ac
            a = max(a, v)
        if v>=b:
            return v, move
    return v, move

def min_valueab(asp: AdversarialSearchProblem[GameState, Action], state, a, b, player):
    if asp.is_terminal_state(state):
        return asp.evaluate_terminal(state)[player], None
    v = sys.maxsize**1000
    move = None
    for ac in asp.get_available_actions(state):
        v2, ac2 = max_valueab(asp, asp.transition(state, ac), a, b, player)
        if v2<v:
            v, move = v2, ac
            b = min(b, v)
        if v<=a:
            return v, move
    return v, move

def alpha_beta_cutoff(
    asp: AdversarialSearchProblem[GameState, Action],
    cutoff_ply: int,
    heuristic_func: Callable[[GameState], float],
) -> Action:
    """
    This function should:
    - search through the asp using alpha-beta pruning
    - cut off the search after cutoff_ply moves have been made.

    Input:
        asp - an AdversarialSearchProblem
        cutoff_ply - an Integer that determines when to cutoff the search and
            use heuristic_func. For example, when cutoff_ply = 1, use
            heuristic_func to evaluate states that result from your first move.
            When cutoff_ply = 2, use heuristic_func to evaluate states that
            result from your opponent's first move. When cutoff_ply = 3 use
            heuristic_func to evaluate the states that result from your second
            move. You may assume that cutoff_ply > 0.
        heuristic_func - a function that takes in a GameState and outputs a
            real number indicating how good that state is for the player who is
            using alpha_beta_cutoff to choose their action. You do not need to
            implement this function, as it should be provided by whomever is
            calling alpha_beta_cutoff, however you are welcome to write
            evaluation functions to test your implemention. The heuristic_func
            we provide does not handle terminal states, so evaluate terminal
            states the same way you evaluated them in the previous algorithms.
    Output:
        an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    player = asp._start_state.player_to_move()
    value, move = max_valueabc(asp, asp.get_start_state(), -sys.maxsize**1000, sys.maxsize**1000, player, 0, cutoff_ply)
    return move

def max_valueabc(asp: AdversarialSearchProblem[GameState, Action], state, a, b, player, depth, cutoff_ply):
    if asp.is_terminal_state(state):
        return asp.evaluate_terminal(state)[player], None
    if depth >= cutoff_ply:
        return asp.heuristic_func(state, player), None
    v = -sys.maxsize**1000
    move = None
    for ac in asp.get_available_actions(state):
        v2, ac2 = min_valueabc(asp, asp.transition(state, ac), a, b, player, depth +1, cutoff_ply)
        if v2>v:
            v, move = v2, ac
            a = max(a, v)
        if v>=b:
            return v, move
    return v, move

def min_valueabc(asp: AdversarialSearchProblem[GameState, Action], state, a, b, player, depth, cutoff_ply):
    if asp.is_terminal_state(state):
        return asp.evaluate_terminal(state)[player], None
    if depth >= cutoff_ply:
        return asp.heuristic_func(state, player), None
    v = sys.maxsize**1000
    move = None
    for ac in asp.get_available_actions(state):
        v2, ac2 = max_valueabc(asp, asp.transition(state, ac), a, b, player, depth +1, cutoff_ply)
        if v2<v:
            v, move = v2, ac
            b = min(b, v)
        if v<=a:
            return v, move
    return v, move

