# Softmax Activation Function Implementation (easy)
# https://www.deep-ml.com/problem/Softmax%20Activation%20Function%20Implementation
# Write a Python function that computes the softmax activation for a given list of scores. The function should return the softmax values as a list, each rounded to four decimal places.
'''
Example:
        input: scores = [1, 2, 3]
        output: [0.0900, 0.2447, 0.6652]
        reasoning: The softmax function converts a list of values into a probability distribution. The probabilities are proportional to the exponential of each element divided by the sum of the exponentials of all elements in the list.
'''

import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    e_max = max(scores)
    e_list = []
    for score in scores:
        e_list.append(math.exp(score - e_max))
    deno = sum(e_list)

    probabilities = []
    for e_value in e_list:
        probabilities.append(round(e_value / deno, 4))
    return probabilities

