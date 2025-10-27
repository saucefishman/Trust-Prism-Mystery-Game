from puzzle import generate_puzzle, generate_puzzles_recursive


def main():
    puzzles = generate_puzzles_recursive(0, 5)
    for puzzle in puzzles:
        pw = puzzle.get_potential_worlds()
        print(pw.num_potential_murderers(), puzzle.estimate_alpha_beta())

if __name__ == '__main__':
    main()