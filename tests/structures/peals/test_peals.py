from sympy.combinatorics import Permutation
from music.structures.peals import Peals, print_peal


# Test sulla classe Peals
def test_transpositions_peal():
    peals = Peals()
    permutation = Permutation([0, 1, 2, 3])
    peals.transpositions_peal(permutation, peal_name="test_peal")
    assert peals.peals["test_peal"]


def test_twenty_all_over():
    # TODO: add test cases
    pass


def test_an_eight_and_forty():
    # TODO: add test cases
    pass


# Test sulla funzione print_peal
def test_print_peal(capsys):
    peal = [[0, 1, 2, 3], [3, 2, 1, 0]]
    print_peal(peal, hunts=[0, 1])
    captured = capsys.readouterr()
    assert captured.out == '\x1b[33;47m0\x1b[0m\x1b[32;47m1\x1b[0m' \
        '\x1b[36;47m2\x1b[0m\x1b[35;47m3\x1b[0m\n\x1b[35;47m3\x1b[0m' \
        '\x1b[36;47m2\x1b[0m\x1b[32;47m1\x1b[0m\x1b[33;47m0\x1b[0m\n'
