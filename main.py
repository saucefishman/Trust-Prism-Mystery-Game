from puzzle import generate_puzzle

def main():
    puzzle = generate_puzzle(1.0, 0.5, 30)

    print('==== FINAL ==== ')

    print(puzzle.world.get_suspect_names())
    print(puzzle.get_potential_worlds().num_potential_murderers())

    for clue in puzzle.clues:
        print(clue.describe())

if __name__ == '__main__':
    main()