import datetime
from logging import getLogger

from anyblok import Declarations
from anyblok.column import String, Integer, Boolean, DateTime
from anyblok.relationship import Many2One, One2One

from anyblok_ai.bloks.ml_models.prediction_models import PredictionModel

logger = getLogger(__name__)

Model = Declarations.Model
Mixin = Declarations.Mixin


@Declarations.register(Model)
class PredictionInputVector(Mixin.IdColumn):
    """Inputs that were sent to a model"""


@Declarations.register(Model)
class PredictionModelInput(Mixin.IdColumn):
    """Sets of inputs for a model"""
    order = Integer(label='Number in the inputs vector', nullable=False)
    name = String(label='Name of the input', nullable=False)
    is_internal_feature = Boolean(
        label='Is internal feature',
        default=True,
        nullable=False
    )
    given_value = String(label='Possible given value')
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
        backref='inputs',
        nullable=False
    )
    prediction_output = String(nullable=False)

    def __repr__(self):
        msg = '<PredictionModelCall call_datetime={self.call_datetime}, prediction_model={self.prediction_model}>'
        return msg.format(self=self)


@Declarations.register(Model)
class PredictionModel:
    """We override the predict from PredictionModel to use the new predict_and_keep_track
    """
    def predict(self, features):
        return self.model_executor.predict_and_keep_track(features)


@Declarations.register(Model)
class PredictionModelExecutor:
    def predict_and_keep_track(self, features):
        current_model = self.prediction_model
        input_vec = self.registry.PredictionInputVector.insert()
        for i, feature in enumerate(features):
            self.registry.PredictionModelInput.insert(
                order=i,
                name=feature['name'],
                given_value=feature['value'],
                prediction_model=current_model,
                input_vector=input_vec
            )

        output = self.predict(features)

        self.registry.PredictionModelCall.insert(
            prediction_model=current_model,
            prediction_inputs=input_vec,
            prediction_output=output,
        )
        return output
