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
            input_order=0,
            input_name='size'
        )
        input_b = registry.PredictionModelInput.insert(
            prediction_model=p_model,
            input_order=1,
            input_name='colour'
        )
        assert registry.PredictionModelInput.query().count() == inputs_count + 2
        assert input_a.input_order == 0
        assert input_b.input_order == 1
        assert p_model.model_inputs == [input_a, input_b]


@freeze_time('2019-10-31')
class TestModelCall:
    def test_create_model_call(self, rollback_registry):
        registry = rollback_registry
        p_model = registry.PredictionModel.insert(
            model_name='dumb model',
            model_file_path=''
        )
        p_model_call_count = registry.PredictionModelCall.query().count()
        p_model_call = registry.PredictionModelCall.insert(
            prediction_model=p_model,
            prediction_output='result'
        )
        assert registry.PredictionModelCall.query().count() == p_model_call_count + 1
        assert p_model_call.prediction_output == 'result'