"""Prediction model
"""
import pickle
from logging import getLogger

from anyblok import Declarations
from anyblok.column import String, Text

logger = getLogger(__name__)

Model = Declarations.Model
Mixin = Declarations.Mixin


@Declarations.register(Model)
class PredictionModel(Mixin.IdColumn, Mixin.TrackModel):
    """PredictionModel"""
    model_name = String(label='Model name', unique=True, nullable=False)
    model_file_path = Text(
        label='Serialized model path',
        nullable=False
    )

    def __str__(self):
        return 'Model {} at {}'.format(self.model_name, self.model_file_path)

    def __repr__(self):
        msg = '<PredictionModel: model_name={self.model_name}, model_file_path={self.model_file_path}>'
        return msg.format(self=self)


def predict(current_model, features):
    """
    Function attached to PredictionModel as does the prediction
    and conserve inputs given+output generated in the DB
    :param current_model:
    :param features:
    :return:
    """
    # todo: add more possibilities: tensorflow, h2o
    logger.info('Starting prediction for {}'.format(current_model.model_name))
    logger.debug('Loading model from {}'.format(current_model.model_file_path))
    with open(current_model.model_file_path, 'rb') as f:
        model = pickle.load(f)

    feature_values = [f['value'] for f in features]
    output = model.predict(feature_values)

    input_vec = current_model.registry.PredictionInputVector.insert()
    for i, feature in enumerate(features):
        current_model.registry.PredictionModelInput.insert(
            input_order=i,
            input_name=feature['name'],
            prediction_model=current_model,
            input_vector=input_vec
        )
    current_model.registry.PredictionModelCall.insert(
        prediction_model=current_model,
        prediction_inputs=input_vec,
        prediction_output=output,
    )
    return output

setattr(PredictionModel, 'predict', predict)
