# Supreme Court ML Prediction Project
The project uses historic United States Supreme Court cases to train natural language processing models to predict case rulings.

### Project Assumptions and Things to Know
1. The number of unique roles within the advocates' file is too numerous to be helpful, so we merged them into 5 categories. While this merger may remove some variability and nuance in the file, we believe it will make it easier to derive meaningful conclusions.
   - The groupings for the roles are as follows: `inferred`, `for respondent`, `for partitioner`, and `for amicus curiae`
   - The code for that grouping can be found in the `clean_roles` function in the [`descriptives.py` file.](https://github.com/michplunkett/supreme-court-ml-predictions/blob/main/supreme_court_predictions/statistics/descriptives.py)
2. The years included within this data set are 2014 to 2019.
3. The datasets included within the previously mentioned year range are ones where the winnings side was either 0 or 1 (no missing, etc.).
4. All reports and presentations are contained within the `reports` folder in the base directory.
   - The Jupyter notebooks cannot be run from the `reports` directory as they use paths that require being in the base directory. If you want to run them yourself, move them to the base directory.

### Project Requirements
- Python version: `^3.11`
- [Poetry](https://python-poetry.org/)

### Technical Notes
- Any modules should be added via the `poetry add [module]` command.
  - Example: `poetry add pytest`

### How do we run this thing?
There are two ways that you can run this application, one of them is to run all components of it at once and the other is to run each component individually. I will give you the instructions for both methods below.

#### Run with One Command
1. After you have installed [Poetry](https://python-poetry.org/docs/basic-usage/), run the command from the base repository directory: `poetry shell`
2. Run the `poetry install` command to install the package dependencies within the project.
3. Run the `make run` command to run the entirety of the application from end-to-end.

#### Run Individual Application Components
1. After you have installed [Poetry](https://python-poetry.org/docs/basic-usage/), run the command from the base repository directory: `poetry shell`
2. Run the command `poetry install` to install the package dependencies within the project.
3. Run the `make get-data` command to get the data from [Convokit](https://convokit.cornell.edu/documentation/supreme.html).
4. Run the `make prepare-data` command to process the cleaned [Convokit](https://convokit.cornell.edu/documentation/supreme.html), produce an Excel sheet containing descriptive statistics of the cleaned data, and prepare the cleaned data to be processed by the machine learning models.
5. Run the `make run-all-models` command to run the Logistic Regression, Random Forest, and XGBoost models on the output of the `prepare-data` command.

#### Repository Health Commands
- `make format`: Runs `Black` on the codebase.
- `make lint`: Runs `ruff` on the codebase.
- `make test`: Runs test cases in the `test` directory.
- `make test-and-fail` Runs test cases in the `test` directory with the `-x` flag that causes a build to fail if a test fails.

## Package Descriptions
- `api`: Houses all functions used to access external APIs.
- `processing`: Houses all functions used to clean and prepare data for statistical analysis and machine learning models.
- `summary_analysis`: Houses all functions used to create our cursory statistical analysis.
- `models`: Houses all functions used to create and run our machine learning models.
- `util`: Houses all functions and constants utilized in multiple packages to prevent code duplication throughout the `supreme-court-predictions` application.
