"""Prediction model
"""
import datetime
import pickle

from anyblok import Declarations, registry
from anyblok.column import String, Integer, DateTime, Boolean, Text

from logging import getLogger

from anyblok.relationship import Many2One, Many2Many, One2Many, One2One

logger = getLogger(__name__)

Model = Declarations.Model
Mixin = Declarations.Mixin

"""
what does a model do?
- it gets created and a pipeline get uploaded
- it gets called
    - it fetches the inputs
    - it call the pipeline with inputs
    - it returns the results of the pipeline
- it can be updated
    - a new pipeline is uploaded
- it can be removed
"""


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


@Declarations.register(Model)
class PredictionInputVector(Mixin.IdColumn):
    """Inputs that were sent to a model"""


@Declarations.register(Model)
class PredictionModelInput(Mixin.IdColumn):
    """Sets of inputs for a model"""
    input_order = Integer(label='Number in the inputs vector', nullable=False)
    input_name = String(label='Name of the input', nullable=False)
    is_internal_feature = Boolean(
        label='Is internal feature',
        default=True,
        nullable=False
    )
    prediction_model = Many2One(
        label='Model to feed with',
        model=PredictionModel,
        nullable=False,
        one2many='model_inputs'
    )
    input_vector = Many2One(
        label='Input group used to predict',
        model=PredictionInputVector,
    )

    def __repr__(self):
        msg = '<PredictionModelInput: ' \
              'input_order={self.input_order}>, input_name={self.input_name}, ' \
              'prediction_model={self.prediction_model}>'
        return msg.format(self=self)


@Declarations.register(Model)
class PredictionModelCall(Mixin.IdColumn):
    """One call to a PredictionModel"""
    call_datetime = DateTime(
        label='Datetime when the model was called',
        default=datetime.datetime.utcnow,
        nullable=False
    )
    prediction_model = Many2One(
        label='Model that created the call',
        model=PredictionModel,
        nullable=False,
        one2many='model_previous_calls'
    )
    prediction_inputs = One2One(
        label='Inputs that produced the output',
        model=PredictionInputVector,
        backref='inputs'
    )
    prediction_output = String(nullable=False)

    def __repr__(self):
        msg = '<PredictionModelCall call_datetime={self.call_datetime}, prediction_model={self.prediction_model}>'
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
