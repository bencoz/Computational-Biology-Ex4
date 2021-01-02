import sys
from viterbi import viterbi


def run_algorithem(algorithm, sequence, parameters):
    # TODO:: Build transition/emission matrices from parameters and call
    if algorithm == 'V':
        # viterbi(sequence)
        pass
    elif algorithm == 'B':
        pass


if __name__ == "__main__":
    # Check if input is valid
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
        parameters.append(sys.argv[i])

    print(" ".join(sequence))
    print('-' * len(sequence) * 2)
    run_algorithem(algorithm, sequence, parameters)
    print('\nDone.')
