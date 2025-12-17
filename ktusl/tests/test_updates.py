from ktusl.models.ktusl import KTUSL
from ktusl.models.bkt import BKT

def test_ktusl_update():
    m = KTUSL(a=0.5, W=2.0)
    m.update(1, [10], 1)  # one success
    p = m.predict_concept_proba(1, 10)
    assert 0.5 < p < 1.0

def test_bkt_update():
    m = BKT(p_init=0.2, p_learn=0.1, p_guess=0.2, p_slip=0.1)
    m.update(1, [10], 1)
    p = m.predict_concept_proba(1, 10)
    assert 0.2 <= p <= 1.0
