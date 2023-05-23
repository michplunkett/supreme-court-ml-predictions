# Supreme Court ML Prediction Project
The project uses historic United States Supreme Court cases to train natural language processing models to predict case rulings.

### Project Assumptions and Things to Know
1. The number of unique roles within the advocates' file is too numerous to be helpful, so we merged them into 5 categories. While this merger may remove some variability and nuance in the file, we believe it will make it easier to derive meaningful conclusions.
   - The groupings for the roles are as follows: `inferred`, `for respondent`, `for partitioner`, and `for amicus curiae`
   - The code for that grouping can be found in the `clean_roles` function in the [`descriptives.py` file.](https://github.com/michplunkett/supreme-court-ml-predictions/blob/main/supreme_court_predictions/statistics/descriptives.py)
2. The years included within this data set are 2014 to 2019.
3. The datasets included within the previously mentioned year range are ones where the winnings side was either 0 or 1 (no missing, etc.).

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
<<<<<<< HEAD
4. Run the `make clean-data` command to clean the data so that we can have our data in the format needed for the downstream functions.
5. Run the `make describe-data` command to process the cleaned [Convokit](https://convokit.cornell.edu/documentation/supreme.html) and produce an Excel sheet that contains basic descriptive statistics of the data.

[THIS WILL BE UPDATED ONCE ALL OF OUR STEPS ARE IN PLACE]

## Standard Commands
- `make format`: Runs `Black` on the codebase
- `make lint`: Runs `ruff` on the codebase
- `make test`: Runs test cases in the `test` directory
- `make run`: Runs the `main` function in the `supreme-court-predictions` folder
- `make get-data`: This function gets the initial data from [Convokit](https://convokit.cornell.edu/documentation/supreme.html)
- `make clean-data`: This function cleans the downloaded [Convokit](https://convokit.cornell.edu/documentation/supreme.html) and provides tokenizations of utterances data.
- `make describe-data`: This function parses the cleaned [Convokit](https://convokit.cornell.edu/documentation/supreme.html) data and produces an Excel file that contains basic descriptive statistics of the data.
- `make process-data`: This function parses the tokenized [Convokit](https://convokit.cornell.edu/documentation/supreme.html) data and produces dataframes collapsing tokenization by case IDs.
- `make logistic-regression`: This function runs a logistic regression on the output of the `process-data` functionality.

[THIS WILL BE UPDATED ONCE ALL OF OUR STEPS ARE IN PLACE]

## Package Descriptions

Within the `api` directory, the `client.py` file for convokit downloads initial data through the get_data function, and the google_cloud_platform `client.py` file will send computing jobs to the cloud. 

In the `util` directory, the `constants.py` file declares often used constants.  

The `data` directory will contain the data for the initial download in convokit, a cleaned version in clean_convokit, results from statistical summaries in statistics, and model analysis results in models. 

In the `statistics` directory, `service.py` cleans data and provides statical summaries using `datacleaner.py` and `descriptives.py`. 

Specifically, `datacleaner.py` provides a DataCleaner class that provides functionality to download, load, and clean convokit data. `parse_all_data` method can clean and parse the Supreme Court Corpus data, convert data into pandas DataFrames, and save the cleaned data to CSV files in `/supreme_court_predictions/data/clean_convokit`.

`descriptives.py` provides a Descriptives class that generates summary statistics on cases, advocates, speakers, voters, and utterances. The `parse_all_data` method generates these summary statistics and exports data as separate CSV files and one complete Excel file. 
=======
4. Run the `make prepare-data` command to process the cleaned [Convokit](https://convokit.cornell.edu/documentation/supreme.html), produce an Excel sheet containing descriptive statistics of the cleaned data, and prepare the cleaned data to be processed by the machine learning models.
5. Run the `make run-all-models` command to run the Logistic Regression, Random Forest, and XGBoost models on the output of the `prepare-data` command.

#### Individual commands
- `make format`: Runs `Black` on the codebase.
- `make lint`: Runs `ruff` on the codebase.
- `make test`: Runs test cases in the `test` directory.
- `make test-and-fail` Runs test cases in the `test` directory with the `-x` flag that causes a build to fail if a test fails.
- `make run`: Runs the entirety of the `supreme-court-predictions` application from end-to-end.
- `make get-data`: Gets the initial data from [Convokit](https://convokit.cornell.edu/documentation/supreme.html).
- `make prepare-data`: Runs the `clean-data`, `describe-data`, `tokenize-data`, and `process-data` `make` commands.
- `make run-all-models`: Runs the `logistic-regression`, `random-forest`, and `xg-boost` `make` commands.
- `make clean-data`: Cleans the downloaded [Convokit](https://convokit.cornell.edu/documentation/supreme.html) and provides tokenizations of utterances data.
- `make describe-data`: Parses the cleaned [Convokit](https://convokit.cornell.edu/documentation/supreme.html) data and produces an Excel file that contains basic descriptive statistics of the data.
- `make tokenize-data`: Tokenizes the cleaned [Convokit](https://convokit.cornell.edu/documentation/supreme.html) corpus DataFrames.
- `make process-data`: Parses the tokenized [Convokit](https://convokit.cornell.edu/documentation/supreme.html) data and produces dataframes collapsing tokenization by case IDs.
- `make logistic-regression`: Runs the [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) model on the tokenized [Convokit](https://convokit.cornell.edu/documentation/supreme.html) data.
- `make random-forest`: Runs the [Random Forest](https://en.wikipedia.org/wiki/Random_forest) model on the tokenized [Convokit](https://convokit.cornell.edu/documentation/supreme.html) data.
- `make xg-boost`: Runs the [XGBoost](https://en.wikipedia.org/wiki/XGBoost) model on the tokenized [Convokit](https://convokit.cornell.edu/documentation/supreme.html) data.

## Package Descriptions
- `api`: Houses all functions used to access external APIs.
- `processing`: Houses all functions used to clean and prepare data for statistical analysis and machine learning models.
- `summary_analysis`: Houses all functions used to create our cursory statistical analysis.
- `models`: Houses all functions used to create and run our machine learning models.
- `util`: Houses all functions and constants utilized in multiple packages to prevent code duplication throughout the `supreme-court-predictions` application.
>>>>>>> main
