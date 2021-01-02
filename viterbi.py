import numpy as np
import math
import sys
from constants import sequence, transition_matrix, emission_matrix, emission_matrix2
from utils import mylog


def viterbi(s, transitions, emissions):
    s_length = len(s)  # n.Rows
    num_of_states = len(emissions)  # k.Columns

    v = np.zeros((num_of_states, s_length), dtype=object)
    v[0, 0] = (math.log(1), -1)  # the tuple is to know from what i value in the previous column the maximum was chosen.
    # initialize v[0, j]
    for i in range(1, num_of_states):
        v[i, 0] = (math.log(emissions[0].get(s[0])), -1)  # there is no previous because this is the most left column.

    for i in range(1, len(s)):
        for j in range(0, num_of_states):
            curr_max = -math.inf
            max_prev_state_index = -1
            emission = emissions[j].get(s[i])
            for l in range(0, num_of_states):
                score = mylog(emission) + float(v[l, i - 1][0]) + mylog(transitions[l, j])

                if score > curr_max:
                    curr_max = score
                    max_prev_state_index = l
            v[j, i] = (curr_max, max_prev_state_index)

    last_column_max = -math.inf
    result = []
    # Find the max in the last column
    prev_index = -1
    for idx in range(num_of_states):
        if v[idx, len(s) - 1][0] > last_column_max:
            last_column_max = v[idx, len(s) - 1][0]
            prev_index = idx

    # Reconstructing
    for k in reversed(range(0, len(s))):
        result.append((
            s[k], prev_index, v[prev_index, k][0]
        ))
        prev_index = v[prev_index, k][1]

    result.reverse()

    return result


result = viterbi(sequence, transition_matrix, emission_matrix)
result_h = viterbi(sequence, transition_matrix, emission_matrix2)

print("\n************************************* section (d) *******************************************\n")

"""  (d)  """
print("Base\t|\tState\t|\tProb")
for index in range(len(sequence)):
    print(result[index][0], "\t\t|\t", result[index][1] + 1, "\t\t|\t", result[index][2])

print("\n************************************* section (h) *******************************************\n")

"""  (h)  """
print("Base\t|\tState\t|\tProb")
for index in range(len(sequence)):
    print(result_h[index][0], "\t\t|\t", result_h[index][1] + 1, "\t\t|\t", result_h[index][2])

a=""
for index in range(len(sequence)):
    a+=str(result_h[index][1] + 1)
    a += "-"

print("\n",a)
print("\n***********************************************************************************")