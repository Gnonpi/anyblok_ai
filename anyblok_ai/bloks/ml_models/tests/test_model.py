import pickle

from freezegun import freeze_time


class TestPredictionModel:
    def test_create_prediction_model(self, rollback_registry):
        registry = rollback_registry
        p_model_count = registry.PredictionModel.query().count()
        dumb_model = registry.PredictionModel.insert(
            model_name='dumb prediction model 01',
            model_file_path='',
        )
        assert registry.PredictionModel.query().count() == p_model_count + 1
        assert dumb_model.model_name == 'dumb prediction model 01'


# defined here to be pickle-able
class FakeModel:
    def predict(self, data):
        return 10.


@freeze_time('2019-10-31')
class TestModelPredict:
    def test_model_predict_create_instances(self, tmpdir, rollback_registry):
        registry = rollback_registry
        model_path = tmpdir / 'test-model.pkl'
        expected_output = 10.

        with open(model_path, 'wb') as f:
            pickle.dump(FakeModel(), f)

        p_model = registry.PredictionModel.insert(
            model_name='dumb model',
            model_file_path=str(model_path)
        )
        p_model_call_count = registry.PredictionModelCall.query().count()
        p_model_input_a = registry.PredictionModelInput.insert(
            prediction_model=p_model,
            input_order=0,
            input_name='size'
        )
        p_model_input_b = registry.PredictionModelInput.insert(
            prediction_model=p_model,
            input_order=1,
            input_name='colour'
        )
        features = [
            {'name': 'size', 'value': 0.5},
            {'name': 'colour', 'value': 'Blue'}
        ]
        output = p_model.predict(features)

        assert registry.PredictionModelCall.query().count() == p_model_call_count + 1
        p_model_call = registry.PredictionModelCall.query().filter_by(
            prediction_model=p_model,
        )[0]
        p_model_inputs = registry.PredictionInputVector.query().all()[0]
        assert p_model_call.prediction_output == expected_output
        assert p_model_call.prediction_inputs == p_model_inputs

        assert output == expected_output
