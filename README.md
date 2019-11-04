# Blok: mlmodels

A Blok that allows to add/update/delete/use machine learning models

## Installation

Make sure to have the `pyproject.toml` for the dev dependencies and all,
and the `setup.py` that link the package to AnyBlok.
I'm not 100% sure why the `setup.py` is mandatory.

Then just: 
```
$ poetry install
```

If you wanna install for production:
```
$ poetry install --no-dev
```

## Running tests

```
$ poetry run pytest
```

## TODO

* [ ] f
* [ ] f
* [ ] f
* [ ] f
* [ ] f
* [ ] f

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
$ poetry run python setup.py install
$ poetry run anyblok_updatedb -c test.cfg
$ poetry run pytest
```

If you made changes to your models
(maybe just add models?),
you have to relaunch:
```
$ poetry run python setup.py install
```

--- 
When using the blok in an application,
remember to:
* install the package in the virtualenv used
* anyblok_updatedb
* make run-dev 