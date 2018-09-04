"""
Python file for testing crossovers and recombinations
"""
import numpy as np
import random
import warnings

from collections import Sequence
from itertools import repeat


def partially_mapped_crossover(p1, p2):
    """
    Function to do a partially mapped crossover on a
    pair of sequenced individuals
    :param p1:
    :param p2:
    :return:
    """
    # Initialize offspring from parents
    size = min(len(p1), len(p2))
    offspring1, offspring2 = [0]*size, [0]*size
    for i in range(size):
        offspring1[p1[i]] = i
        offspring2[p2[i]] = i

    print(offspring1)
    print(offspring2)



def main():
    #Initialize population
    P1 = [2, 4, 7, 1, 3, 6, 8, 9, 5]
    P2 = [5, 9, 8, 6, 2, 4, 1, 3, 7]
    partially_mapped_crossover(P1, P2)

main()