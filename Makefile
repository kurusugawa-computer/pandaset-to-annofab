ifndef TARGET
	export TARGET:=panda2anno
endif

.PHONY: init lint format test
init:
	pip install poetry --upgrade
	poetry install

format:
	poetry run black .
	poetry run autoflake  --in-place --remove-all-unused-imports  --ignore-init-module-imports --recursive ${TARGET}
	poetry run isort ${TARGET}

lint:
	poetry run mypy ${TARGET} tests --config-file setup.cfg
	poetry run flake8 ${TARGET}
	poetry run pylint ${TARGET} --rcfile setup.cfg

test:
	poetry run pytest -n auto  --cov=panda2anno --cov-report=html tests

