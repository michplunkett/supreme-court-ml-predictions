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

# Group level commands
.PHONY: get-data
get-data:
	python -m ${BASEDIR} --get-data

.PHONY: prepare-data
prepare-data:
	python -m ${BASEDIR} --clean-data --describe-data --tokenize-data --process-data

.PHONY: run-all-models
run-all-models:
	python -m ${BASEDIR} --logistic-regression --random-forest --xg-boost --simulate

.PHONY: run
run:
	make get-data clean-data prepare-data run-all-models

# Individual function commands
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

.PHONY: logistic-regression
logistic-regression:
	python -m ${BASEDIR} --logistic-regression

.PHONY: random-forest
random-forest:
	python -m ${BASEDIR} --random-forest

.PHONY: xg-boost
xg-boost:
	python -m ${BASEDIR} --xg-boost

.PHONY: simulate
simulate:
	python -m ${BASEDIR} --simulate
