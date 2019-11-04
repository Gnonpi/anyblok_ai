# Blok: mlmodels

A Blok that allows to add/update/delete/use machine learning models

## TODO

Mandatory:
* [ ] Make the PredictionModels.predict method work
* [ ] Make the predict method create in/out records with ml_features
* [ ] Test everything

Good to have:
* [ ] More ML libraries supported
* [ ] Mechanism to add models: link executable
* [ ] Remove mandatory scikit-learn dependency


### To make everything work

A postgres db in docker:
```
$ docker run --rm --name pg-docker  -e POSTGRES_DB=anyblok_mlmodels_test -p 5432:5432 postgres
```

Then the config in `test.cfg`:
```
[AnyBlok]
db_host=localhost
db_name=anyblok_mlmodels_test
db_user_name=postgres
db_port=5432
db_driver_name=postgresql
install_or_update_bloks=mlmodels

logging_configfile = logging.cfg
```

Then update the DB with the mlmodels blok:
```
$ python setup.py install
$ anyblok_updatedb -c test.cfg
$ pytest
```

If you made changes to your models
(maybe just add models?),
you have to relaunch:
```
$ python setup.py install
```

When adding a new blok in your package:
* add the new blok code in the package dunder-init, models, tests, etc...
* add the entrypoint in the `setup.py`
* in the test.cfg, add the name to `install_or_update_bloks` entry.
The name to add is the name of the entrypoint, not the package

--- 
When using the blok in an application,
remember to:
* install the package in the virtualenv used
* anyblok_updatedb
* make run-dev 