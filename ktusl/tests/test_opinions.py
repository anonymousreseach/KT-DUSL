from ktusl.models.opinions import prior_alpha_beta, expectation

def test_beta_expectation():
    a, W = 0.5, 2.0
    alpha, beta = prior_alpha_beta(a, W)
    assert abs(expectation(alpha, beta) - 0.5) < 1e-9
