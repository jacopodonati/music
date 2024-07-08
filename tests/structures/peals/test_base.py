from music.structures.peals import GenericPeal


# Test sulla classe GenericPeal
def test_act():
    peals = GenericPeal()
    peals.nelements = 4
    peals.peals = {"test_peal": [lambda x: x[::-1]]}
    result = peals.act("test_peal", domain=[0, 1, 2, 3])
    assert result == [[3, 2, 1, 0]]


def test_act_all():
    peals = GenericPeal()
    peals.nelements = 4
    peals.peals = {"test_peal": [lambda x: x[::-1]]}
    peals.act_all(domain=[0, 1, 2, 3])
    assert peals.acted_peals == {"test_peal_acted": [[3, 2, 1, 0]]}
