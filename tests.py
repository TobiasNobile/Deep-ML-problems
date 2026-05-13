from solutions.problem_768.solution import *

def tests():
    assert rejection_sampling_best_of_k([['a','b','c']], [[0.1, 0.5, 0.3]]) == ["b"]