#%%
import itertools
from fractions import Fraction
from math import comb

from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.causal_discovery.local.orientation_using_scg import sOrientability
from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph


def build_scg(p, mask, pairs):
    """Build the SCG on p nodes whose edge set is given by a 0/1 mask over pairs."""
    scg = SummaryCausalGraph()
    scg.add_vertices([str(i) for i in range(p)])
    for keep, (i, j) in zip(mask, pairs):
        if keep:
            scg.add_directed_edge(str(i), str(j))
    return scg


def is_fully_s_orientable(scg, p):
    """True iff every ordered pair i != j is s-orientable."""
    return all(
        sOrientability(scg, DTimeVar(str(i), 0), DTimeVar(str(j), 0)).is_s_orientable()
        for i, j in itertools.permutations(range(p), 2)
    )


def canonical(mask, pairs, idx, perms):
    """Smallest mask over all node relabelings (isomorphism-class key)."""
    best = None
    for perm in perms:
        new = [0] * len(mask)
        for keep, (i, j) in zip(mask, pairs):
            if keep:
                new[idx[(perm[i], perm[j])]] = 1
        t = tuple(new)
        if best is None or t < best:
            best = t
    return best


def enumerate_pi(p, use_symmetry=True):
    """Exact proportion of SCGs on p nodes that are NOT fully s-orientable."""
    pairs = list(itertools.product(range(p), repeat=2))  # includes self-loops
    idx = {pr: n for n, pr in enumerate(pairs)}
    perms = list(itertools.permutations(range(p)))
    cache, n_bad, total = {}, 0, 0
    for mask in itertools.product([0, 1], repeat=len(pairs)):
        total += 1
        key = canonical(mask, pairs, idx, perms) if use_symmetry else mask
        if key not in cache:
            cache[key] = not is_fully_s_orientable(build_scg(p, key, pairs), p)
        n_bad += cache[key]
    return Fraction(n_bad, total)


def closed_form(p):
    """pi(p) = 1 - 2^-p * sum_k C(p,k) prod_{i<k} (1 - i/2^p)."""
    N = 2**p
    s = Fraction(0)
    for k in range(p + 1):
        prod = Fraction(1)
        for i in range(k):
            prod *= 1 - Fraction(i, N)
        s += comb(p, k) * prod
    return 1 - s / N


if __name__ == "__main__":
    for p in [1, 2, 3, 4]:
        pi, cf = enumerate_pi(p), closed_form(p)
        print(f"p={p}: enum={pi} ({float(pi):.5f}) | "
              f"formula={cf} ({float(cf):.5f}) | equal: {pi == cf}")
        
        
        
for p in range(2, 21):
    print(f"p={p}: {float(closed_form(p)):.6f}")
# %%
