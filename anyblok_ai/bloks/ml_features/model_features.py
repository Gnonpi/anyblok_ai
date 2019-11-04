import datetime

from anyblok import Declarations
from anyblok.column import String, Integer, Boolean, DateTime
from anyblok.relationship import Many2One, One2One

from bloks.ml_models.prediction_models import Model, Mixin, PredictionModel


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
