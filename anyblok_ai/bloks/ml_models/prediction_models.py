"""Prediction model
"""
import pickle
from logging import getLogger

from anyblok import Declarations
from anyblok.column import String, Text
from anyblok.relationship import One2One

logger = getLogger(__name__)

Model = Declarations.Model
Mixin = Declarations.Mixin


@Declarations.register(Model)
class PredictionModel(Mixin.IdColumn, Mixin.TrackModel):
    """A stat model takes features in and output a prediction"""

    model_name = String(label='Model name', unique=True, nullable=False)
    model_file_path = Text(
        label='Serialized model path',
        nullable=False
    )  # todo: find better data type

    def predict(self, features):
        return self.model_executor.predict(features)

    def __str__(self):
        return 'Model {} at {}'.format(self.model_name, self.model_file_path)

    def __repr__(self):
        msg = '<PredictionModel: model_name={self.model_name}, model_file_path={self.model_file_path}>'
        return msg.format(self=self)


@Declarations.register(Model)
class PredictionModelExecutor(Mixin.IdColumn):
    """An executor is the part of the model that really compute
    Should be extended or overriden in other bloks
    """
    prediction_model = One2One(
        label='Model using the executor',
        model=PredictionModel,
        backref='model_executor',
        nullable=False
    )

    def predict(self, features):
        """Pass the features to the model, let the model run and return its output"""
        # todo: add more possibilities: tensorflow, h2o
        model_name = self.prediction_model.model_name
        model_file_path = self.prediction_model.model_file_path
        logger.info('Starting prediction for {}'.format(model_name))
        logger.debug('Loading model from {}'.format(model_file_path))
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)

        feature_values = [f['value'] for f in features]
        output = model.predict(feature_values)
        return output
