""" Test for music.tables """
import pytest
import matplotlib.pyplot as plt
from music.tables import PrimaryTables


@pytest.fixture(name="table")
def table_ficture():
    """ Sample data. """
    return PrimaryTables(size=2048)


def test_primary_tables_init(table):
    """ Test table initiation. """
    assert table.size == 2048
    assert table.sine is not None
    assert table.saw is not None
    assert table.square is not None
    assert table.triangle is not None


def test_primary_tables_make_tables(table):
    """ Test table conformity. """
    assert len(table.sine) == 2048
    assert len(table.saw) == 2048
    assert len(table.square) == 2048
    assert len(table.triangle) == 2048


# def test_primary_tables_draw_tables(table):
#     """ Test table plotting. """
#     fig = plt.figure()
#     fig_canvas = fig.canvas
#     table.draw_tables()
#     assert len(fig_canvas.get_renderer().buffer_rgba()) > 0
