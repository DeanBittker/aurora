"""Proposers. Given sequences, make new candidate sequences by mutating them.

This is the "propose" step of the closed loop. The simplest version is random
directed evolution: take a sequence and change one amino acid at a random
position. The loop calls a proposer to make candidates, scores them with the
oracle, keeps the best, and proposes again from those.
"""
from __future__ import annotations

import random

# The 20 standard amino acids. A protein sequence is just a string over these.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def mutate(sequence: str, n_mut: int, rng: random.Random) -> str:
    """Return a copy of `sequence` with exactly n_mut positions changed.

    Substitutions only, so the length stays the same. Each changed position gets
    a different amino acid than it had, so n_mut really is the number of changes.
    """
    chars = list(sequence)
    positions = rng.sample(range(len(chars)), n_mut)
    for pos in positions:
        current = chars[pos]
        choices = [a for a in AMINO_ACIDS if a != current]
        chars[pos] = rng.choice(choices)
    return "".join(chars)


def propose(parents: list[str], n_children: int, n_mut: int,
            rng: random.Random) -> list[str]:
    """For each parent sequence, make n_children mutated variants.

    Returns one flat list of all the children.
    """
    children = []
    for parent in parents:
        for _ in range(n_children):
            children.append(mutate(parent, n_mut, rng))
    return children
