import pytest

from pyciphod.utils.background_knowledge.background_knowledge import BackgroundKnowledge


def test_mandatory_and_forbidden_edges_and_orientations():
    bk = BackgroundKnowledge()

    bk.add_mandatory_edge('A', 'B')
    bk.add_forbidden_edge('C', 'D')
    bk.add_mandatory_orientation('E', 'F')
    bk.add_forbidden_orientation('G', 'H')

    assert ('A', 'B') in bk.get_mandatory_edges()
    assert ('C', 'D') in bk.get_forbidden_edges()
    assert ('E', 'F') in bk.get_mandatory_orientations()
    assert ('G', 'H') in bk.get_forbidden_orientations()


def test_add_edges_from_and_orientations_from_and_non_descendants_from():
    bk = BackgroundKnowledge()
    bk.add_mandatory_edges_from({('X', 'Y'), ('U', 'V')})
    bk.add_forbidden_edges_from({('M', 'N')})
    bk.add_mandatory_orientations_from({('P', 'Q')})
    bk.add_forbidden_orientations_from({('R', 'S')})

    assert ('X', 'Y') in bk.get_mandatory_edges()
    assert ('M', 'N') in bk.get_forbidden_edges()
    assert ('P', 'Q') in bk.get_mandatory_orientations()
    assert ('R', 'S') in bk.get_forbidden_orientations()

    # non_descendants_from should add forbidden orientations for each nd
    bk.add_non_descendants_from('Z', {'A', 'B'})
    nd = bk.get_non_descendants()
    assert 'Z' in nd
    assert nd['Z'] == {'A', 'B'}
    # forbidden orientations should include (A,Z) and (B,Z) per implementation
    f_or = bk.get_forbidden_orientations()
    assert ('Z', 'A') in f_or and ('Z', 'B') in f_or


def test_add_non_descendant_and_remove():
    bk = BackgroundKnowledge()
    bk.add_non_descendant('T', 'S')
    nd = bk.get_non_descendants()
    assert 'S' in nd
    assert 'T' in nd['S']

    # adding non-descendant also adds forbidden orientation (node, non_descendant) in implementation
    f_or = bk.get_forbidden_orientations()
    assert ('T', 'S') in f_or

    # remove_non_descendant should remove the mapping and the forbidden orientation
    bk.remove_non_descendant('S', 'T')
    nd2 = bk.get_non_descendants()
    assert 'S' not in nd2 or 'T' not in nd2.get('S', set())
    # forbidden orientation should no longer include (T,S)
    assert ('T', 'S') not in bk.get_forbidden_orientations()


def test_getters_return_copies():
    bk = BackgroundKnowledge()
    bk.add_mandatory_edge('A', 'B')
    me = bk.get_mandatory_edges()
    me.add(('X', 'Y'))
    # original should not be modified
    assert ('X', 'Y') not in bk.get_mandatory_edges()

    bk.add_non_descendants_from('N', {'L'})
    nd = bk.get_non_descendants()
    nd['N'].add('Z')
    # original non_descendants should not contain 'Z'
    assert 'Z' not in bk.get_non_descendants().get('N', set())
