# Hyperparameter grids (placeholders)
def grid_ktusl():
    for a in [0.4, 0.5, 0.6]:
        for W in [1.0, 2.0, 4.0]:
            yield {"a": a, "W": W}

def grid_bkt():
    for p_init in [0.1, 0.2, 0.3]:
        for p_learn in [0.05, 0.1, 0.15]:
            for p_guess in [0.1, 0.2, 0.3]:
                for p_slip in [0.05, 0.1, 0.2]:
                    yield {"p_init":p_init,"p_learn":p_learn,"p_guess":p_guess,"p_slip":p_slip}
