from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='anyblok_ai',
    version='0.1.0',
    description="A Blok that allows to add/update/delete/use machine learning models",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Denis Viviès",
    author_email="legnonpi@gmail.com",
    classifiers=['License :: OSI Approved',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7'],
    packages=find_packages(),
    install_requires=[
        'anyblok (>=0.22.5,<0.23.0)',
        'anyblok_mixins (>=1.0.0)',
        'psycopg2 (>=2.8.0,<3.0.0)',
    ],
    dependency_links=[],
    include_package_data=True,
    entry_points={
        'bloks': [
            'mlmodels=anyblok_ai.bloks.ml_models:MachineLearningModelBlok',
            'mlfeatures=anyblok_ai.bloks.ml_features:MachineLearningFeaturesBlok'
        ],
    },
    project_urls={},
)
