import pytest
from sklearn import svm
from sklearn import datasets
from joblib import dump

from anyblok.conftest import *  # noqa: F401,F403
# from anyblok_pyramid.conftest import *  # noqa: F401,F403


@pytest.fixture
def ml_model_pickle(tmpdir):
    tmp_model_path = tmpdir / 'test-model.pkl'
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    dump(clf, tmp_model_path)

    yield tmp_model_path
