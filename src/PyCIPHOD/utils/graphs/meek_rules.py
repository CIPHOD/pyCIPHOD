def meek_rule_1(g, X, Y, Z):
    """
    :param X: a vertex in g
    :param Y:  a vertex in g
    :param Z:  a vertex in g
    :return:  True if the conditions of Meek's Rule 1 are satisfied for the triple (X, Y, Z) in g, False otherwise.
     Meek's Rule 1 states that if there is a directed edge from X to Y (X -> Y) and an undirected edge between Y and Z (Y - Z), and there is no edge between X and Z (X and Z are not adjacent), then we can orient the edge between Y and Z as Y -> Z.
    """
    if (X, Y) in g.get_directed_edges() and (
    Y, Z) in g.get_undirected_edges() and Z not in g.get_adjacencies(X):
        return True
    return False


def meek_rule_2(g, X, Y, Z):
    """
    :param X: a vertex in g
    :param Y:  a vertex in g
    :param Z:  a vertex in g
    :return:  True if the conditions of Meek's Rule 2 are satisfied for the triple (X, Y, Z) in g, False otherwise.
     Meek's Rule 2 states that if there is a directed edge from X to Y (X -> Y) and a directed edge from Y to Z (Y -> Z), and there is an undirected edge between X and Z (X - Z), then we can orient the edge between X and Z as X -> Z.
    """
    if (X, Y) in g.get_directed_edges() and (Y, Z) in g.get_directed_edges() and (
    X, Z) in g.get_undirected_edges():
        return True
    return False


def meek_rule_3(g, X, Y, Z, W):
    """
    :param g: a pattern of a CPDAG
    :param X: a vertex in g
    :param Y:  a vertex in g
    :param Z:  a vertex in g
    :return:  True if the conditions of Meek's Rule 3 are satisfied for the triple (X, Y, Z) and vertex W in g, False otherwise.
     Meek's Rule 3 states that if there is a directed edge from X to Y (X -> Y) and a directed edge from Z to Y (Z -> Y), and there is an undirected edge between X and Z (X - Z), and there exists a vertex W such that there are undirected edges between W and Y (W - Y), W and X (W - X), and W and Z (W - Z), then we can orient the edge between X and Z as X -> Z.
    """
    if (X, Y) in g.get_directed_edges() and (
    Z, Y) in g.get_directed_edges() and Z not in g.get_adjacencies(X):
        if (W, Y) in g.get_undirected_edges() and (W, X) in g.get_undirected_edges() and (
                W, Z) in g.get_undirected_edges():
            return True
    return False