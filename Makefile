# Here is some general information on Makefile's so that you can grow this out:
# https://www.gnu.org/software/make/manual/html_node/Introduction.html

BASEDIR="supreme_court_predictions"

.PHONY: format
format:
	isort ${BASEDIR}/ test/
	black ${BASEDIR}/ test/

.PHONY: lint
lint:
	ruff ${BASEDIR}/ test/

.PHONY: test
test:
	pytest -vs test/

.PHONY: test-and-fail
test-and-fail:
	pytest -vsx test/

.PHONY: run
run:
	python -m ${BASEDIR} --get-data --clean-data --describe-data --process-data

.PHONY: get-data
get-data:
	python -m ${BASEDIR} --get-data

.PHONY: clean-data
clean-data:
	python -m ${BASEDIR} --clean-data

.PHONY: describe-data
describe-data:
	python -m ${BASEDIR} --describe-data

.PHONY: logistic-regression
logistic-regression:
	python -m ${BASEDIR} --logistic-regression

.PHONY: process-data
process-data:
	python -m ${BASEDIR} --process-data