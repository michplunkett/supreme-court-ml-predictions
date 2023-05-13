# Here is some general information on Makefile's so that you can grow this out:
# https://www.gnu.org/software/make/manual/html_node/Introduction.html

BASEDIR="supreme_court_predictions"

.PHONY: format
format:
	isort ${BASEDIR}/ test/
	black ${BASEDIR}/ test/ *.ipynb

.PHONY: lint
lint:
	ruff ${BASEDIR}/ test/

.PHONY: test
test:
	pytest -vs test/

.PHONY: test-and-fail
test-and-fail:
	pytest -vsx test/

.PHONY: get-data
get-data:
	python -m ${BASEDIR} --get-data

.PHONY: clean-data
clean-data:
	python -m ${BASEDIR} --clean-data

.PHONY: describe-data
describe-data:
	python -m ${BASEDIR} --describe-data

.PHONY: tokenize-data
tokenize-data:
	python -m ${BASEDIR} --tokenize-data

.PHONY: process-data
process-data:
	python -m ${BASEDIR} --process-data

.PHONY: run-all-models:
run-all-models:
	python -m ${BASEDIR} --logistic-regression --random-forest --xg-boost

.PHONY: logistic-regression
logistic-regression:
	python -m ${BASEDIR} --logistic-regression

.PHONY: random-forest
random-forest:
	python -m ${BASEDIR} --random-forest

.PHONY: xg-boost
random-forest:
	python -m ${BASEDIR} --xg-boost

.PHONY: run
run:
	make clean-data prepare-data run-all-models

.PHONY: prepare-data
prepare-data:
	python -m ${BASEDIR} --clean-data --describe-data --tokenize-data --process-data
