import pytest
import irisclassifier

performance_thresholds = {0.50, 0.75, 0.95}


def test_evaluation():
    i = irisclassifier.IrisClassifier()
    i.ingestion()
    i.segregation()
    i.train()
    res = i.evaluation()
    assert res > 0.75  # this allows me to pass or fail the test


@pytest.mark.parametrize('th', performance_thresholds)
def test_evaluation(th):
    i = irisclassifier.IrisClassifier()
    i.ingestion()
    i.segregation()
    i.train()
    res = i.evaluation()
    assert res > th  # this allows me to pass or fail the test



