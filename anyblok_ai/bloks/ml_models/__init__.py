from anyblok.blok import Blok
from logging import getLogger

logger = getLogger(__name__)


class MachineLearningModelBlok(Blok):
    """Machine Learning Blok
    """
    version = "0.1.0"
    author = "Denis Viviès"
    required = ['anyblok-core', 'anyblok-mixins']

    @classmethod
    def import_declaration_module(cls):
        from . import prediction_models  # noqa

    @classmethod
    def reload_declaration_module(cls, reload):
        from . import prediction_models
        reload(prediction_models)
