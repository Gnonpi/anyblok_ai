import pickle

from freezegun import freeze_time


class TestModelInputs:
    def test_create_model_inputs(self, rollback_registry):
        registry = rollback_registry
        p_model = registry.PredictionModel.insert(
            model_name='dumb model',
            model_file_path=''
        )
        inputs_count = registry.PredictionModelInput.query().count()
        input_a = registry.PredictionModelInput.insert(
            prediction_model=p_model,
            order=0,
            name='size'
        )
        input_b = registry.PredictionModelInput.insert(
            prediction_model=p_model,
            order=1,
            name='colour'
        )
        assert registry.PredictionModelInput.query().count() == inputs_count + 2
        assert input_a.order == 0
        assert input_b.order == 1
        assert p_model.model_inputs == [input_a, input_b]


@freeze_time('2019-10-31')
class TestModelCall:
    def test_create_model_call(self, rollback_registry):
        registry = rollback_registry
        p_model = registry.PredictionModel.insert(
            model_name='dumb model',
            model_file_path=''
        )
        input_vec = registry.PredictionInputVector.insert()
        p_model_call_count = registry.PredictionModelCall.query().count()
        p_model_call = registry.PredictionModelCall.insert(
            prediction_model=p_model,
            prediction_inputs=input_vec,
            prediction_output='result',
        )
        assert registry.PredictionModelCall.query().count() == p_model_call_count + 1
        assert p_model_call.prediction_output == 'result'


# defined here to be pickle-able
class FakeModel:
    def predict(self, data):
        return 10.


class TestPredictionModelKeepTrack:
    def test_prediction_create_instances(self, tmpdir, rollback_registry, mocker):
        registry = rollback_registry
        model_path = tmpdir / 'test-model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(FakeModel(), f)

        expected_output = 10.
        outputs_count = registry.PredictionModelCall.query().count()

        p_model = registry.PredictionModel.insert(
            model_name='dumb model',
            model_file_path=str(model_path)
        )
        p_model_executor = registry.PredictionModelExecutor.insert(
            prediction_model=p_model
        )
        spy_executor = mocker.spy(p_model_executor, 'predict_and_keep_track')
        features = [
            {'name': 'size', 'value': 0.5},
            {'name': 'colour', 'value': 'Blue'}
        ]
        output = p_model.predict(features)

        assert output == expected_output
        spy_executor.assert_called_once_with(features)

        assert registry.PredictionModelCall.query().count() == outputs_count + 1
        model_call = registry.PredictionModelCall.query().filter_by(prediction_model=p_model).first()
        # TODO: freezegun doesnt seem to be working well with sqlalchemy
        # assert model_call.call_datetime == '2019-10-31'
        assert model_call.prediction_output == 10.

        input_vec = model_call.prediction_inputs
        inputs_used = registry.PredictionModelInput.query().filter_by(input_vector=input_vec).all()
        assert len(inputs_used) == 2

        input_a, input_b = inputs_used
        assert input_a.order == 0
        assert input_a.name == 'size'
        assert input_a.given_value == '0.5'

        assert input_b.order == 1
        assert input_b.name == 'colour'
        assert input_b.given_value == 'Blue'
