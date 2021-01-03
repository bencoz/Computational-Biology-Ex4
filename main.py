import sys
from viterbi import viterbi_training
# from .baum_welch import baum_welch
from utils import build_transition_matrix, build_emission_matrix


def run_algorithm(algorithm, sequence, parameters, epsilon):
    transition = build_transition_matrix(*parameters[:2])
    emission = build_emission_matrix(*parameters[2:])

    if algorithm == 'V':
        viterbi_training(sequence, transition, emission, epsilon)
        pass
    elif algorithm == 'B':
        # baum_welch(sequence, transition, emission, epsilon)
        pass


if __name__ == "__main__":
    # Input validation
    algorithms = ['V', 'B']
    arguments_num = len(sys.argv)
    if arguments_num != 11:
        raise Exception("Arguments number is not valid:")
    sequence = str(sys.argv[1])
    algorithm = str(sys.argv[2])
    if algorithm not in algorithms:
        raise Exception("algorithm name is not valid (V/B are valid)")
    parameters = []
    for i in range(3, arguments_num):
        parameters.append(float(sys.argv[i]))

    epsilon = 1e-5
    print(" ".join(sequence))
    print('-' * len(sequence) * 2)
    run_algorithm(algorithm, sequence, parameters, epsilon)
    print('\nDone.')
