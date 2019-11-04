from anyblok.blok import Blok
from logging import getLogger

logger = getLogger(__name__)


class MachineLearningFeaturesBlok(Blok):
    """Features and output given by a model
    """
    version = "0.1.0"
    author = "Denis Viviès"
    required = ['anyblok-core', 'anyblok-mixins']

    @classmethod
    def import_declaration_module(cls):
        from . import model_features  # noqa

    @classmethod
    def reload_declaration_module(cls, reload):
        from . import model_features
        reload(model_features)
