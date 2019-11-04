import pickle

from freezegun import freeze_time


def test_create_prediction_model(rollback_registry):
    registry = rollback_registry
    p_model_count = registry.PredictionModel.query().count()
    dumb_model = registry.PredictionModel.insert(
        model_name='dumb prediction model 01',
        model_file_path='',
    )
    assert registry.PredictionModel.query().count() == p_model_count + 1
    assert dumb_model.model_name == 'dumb prediction model 01'


def test_create_model_executor(rollback_registry):
    registry = rollback_registry
    dumb_model = registry.PredictionModel.insert(
        model_name='dumb prediction model 01',
        model_file_path='',
    )
    p_executor_count = registry.PredictionModelExecutor.query().count()
    model_executor = registry.PredictionModelExecutor.insert(
        prediction_model=dumb_model
    )
    assert registry.PredictionModelExecutor.query().count() == p_executor_count + 1
    assert model_executor.prediction_model == dumb_model


# defined here to be pickle-able
class FakeModel:
    def predict(self, data):
        return 10.


@freeze_time('2019-10-31')
class TestModelPredict:
    def test_model_predict_create_instances(self, tmpdir, rollback_registry, mocker):
        registry = rollback_registry
        model_path = tmpdir / 'test-model.pkl'
        expected_output = 10.

        with open(model_path, 'wb') as f:
            pickle.dump(FakeModel(), f)

        p_model = registry.PredictionModel.insert(
            model_name='dumb model',
            model_file_path=str(model_path)
        )
        p_model_executor = registry.PredictionModelExecutor.insert(
            prediction_model=p_model
        )
        spy_executor = mocker.spy(p_model_executor, 'predict')
        features = [
            {'name': 'size', 'value': 0.5},
            {'name': 'colour', 'value': 'Blue'}
        ]
        output = p_model.predict(features)

        assert output == expected_output
        spy_executor.assert_called_once_with(features)

