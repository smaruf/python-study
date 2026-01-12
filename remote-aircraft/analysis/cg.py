def center_of_gravity(masses, positions):
    return sum(m*p for m, p in zip(masses, positions)) / sum(masses)
