from sympy.combinatorics import Permutation
from music.structures import (
    InterestingPermutations,
    dist,
    transpose_permutation
)


# Test sulla classe InterestingPermutations
def test_get_alternating():
    ip = InterestingPermutations(nelements=4, method="dimino")
    assert ip.alternations
    assert ip.alternations_complement
    assert ip.alternations_by_sizes


def test_get_rotations():
    ip = InterestingPermutations(nelements=4, method="dimino")
    assert ip.rotations


def test_get_mirrors():
    ip = InterestingPermutations(nelements=4, method="dimino")
    assert ip.mirrors
    assert ip.edge_mirrors
    assert ip.vertex_mirrors


def test_get_swaps():
    ip = InterestingPermutations(nelements=4, method="dimino")
    assert ip.swaps
    assert ip.swaps_as_comes
    assert ip.swaps_by_stepsizes
    assert ip.neighbor_swaps


def test_even_odd():
    ip = InterestingPermutations(nelements=4, method="dimino")
    sequence_even = [0, 1, 2, 3]
    sequence_odd = [1, 0, 2, 3]
    assert ip.even_odd(sequence_even) == 'even'
    assert ip.even_odd(sequence_odd) == 'odd'


def test_get_full_symmetry():
    ip = InterestingPermutations(nelements=4, method="dimino")
    assert ip.permutations
    assert ip.permutations_by_sizes


# Test sulla funzione dist
def test_dist():
    swap1 = Permutation([1, 0, 2, 3])  # Supporto: [0, 1]
    swap2 = Permutation([0, 3, 2, 1])  # Supporto: [1, 3]

    assert dist(swap1) == 1, f"Expected 1, but got {dist(swap1)}"
    assert dist(swap2) == 2, f"Expected 2, but got {dist(swap2)}"

    # Altri casi di test con permutazioni diverse
    swap3 = Permutation([2, 0, 1])  # Supporto: [0, 2]
    assert dist(swap3) == 1, f"Expected 1, but got {dist(swap3)}"

    swap4 = Permutation([3, 1, 2, 0])  # Supporto: [0, 3]
    assert dist(swap4) == 1, f"Expected 1, but got {dist(swap4)}"

    swap5 = Permutation([4, 2, 3, 0, 1])  # Supporto: [0, 4]
    assert dist(swap5) == 1, f"Expected 1, but got {dist(swap5)}"


# Test sulla funzione transpose_permutation
def test_transpose_permutation():
    # Caso base
    permutation = Permutation([0, 1, 2, 3])
    transposed_permutation = transpose_permutation(permutation, step=1)
    assert transposed_permutation == Permutation([1, 2, 3, 4]), f"Expected [1, 2, 3, 4], but got {transposed_permutation}"

    # Test con step negativo
    transposed_permutation = transpose_permutation(permutation, step=-1)
    assert transposed_permutation == Permutation([-1, 0, 1, 2]), f"Expected [-1, 0, 1, 2], but got {transposed_permutation}"

    # Test con step zero (deve restituire la permutazione originale)
    transposed_permutation = transpose_permutation(permutation, step=0)
    assert transposed_permutation == permutation, f"Expected {permutation}, but got {transposed_permutation}"

    # Test con step maggiore della lunghezza della permutazione
    transposed_permutation = transpose_permutation(permutation, step=5)
    assert transposed_permutation == Permutation([5, 6, 7, 8]), f"Expected [5, 6, 7, 8], but got {transposed_permutation}"

    # Test con un'altra permutazione
    permutation = Permutation([2, 0, 1])
    transposed_permutation = transpose_permutation(permutation, step=2)
    assert transposed_permutation == Permutation([4, 2, 3]), f"Expected [4, 2, 3], but got {transposed_permutation}"
