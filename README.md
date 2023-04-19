# Supreme Court ML Prediction Project
The project uses historic supreme court cases to train natural language processing models to predict case rulings.

### Project Requirements
- Python version: `^3.11`
- [Poetry](https://python-poetry.org/)

### Technical Notes
- Any modules should be added via the `poetry add [module]` command.
  - Example: `poetry add pytest`

### How do we run this thing?
There are two ways that you can run this application, one of them is to run all components of it at once and the other is to run each component individually. I will give you the instructions for both methods below.

#### Run with one command
1. After you have installed [Poetry](https://python-poetry.org/docs/basic-usage/), run the command from the base repository directory: `poetry shell`
2. Run the `poetry install` command to install the package dependencies within the project.
3. Run the `make run` command to run the application.

#### Run each function individually
1. After you have installed [Poetry](https://python-poetry.org/docs/basic-usage/), run the command from the base repository directory: `poetry shell`
2. Run the command `poetry install` to install the package dependencies within the project.
3. Run the `make get-data` command to get the data from [Convokit](https://convokit.cornell.edu/documentation/supreme.html).
4. Run the `make clean-data` command to clean the data so that we can have our data in the format needed for the downstream functions.
5. Run the `make describe-data` command to process the cleaned [Convokit](https://convokit.cornell.edu/documentation/supreme.html) and produce an Excel sheet that contains basic descriptive statistics of the data. 

## Standard Commands
- `make format`: Runs `Black` on the codebase
- `make lint`: Runs `pytlint` on the codebase
- `make test`: Runs test cases in the `test` directory
- `make run`: Runs the `main` function in the `supreme-court-predictions` folder
- `make get-data`: This function gets the initial data from [Convokit](https://convokit.cornell.edu/documentation/supreme.html)
- `make clean-data`: This function cleans the downloaded [Convokit](https://convokit.cornell.edu/documentation/supreme.html)
- `make describe-data`: This function parses the cleaned [Convokit](https://convokit.cornell.edu/documentation/supreme.html) data and produces an Excel file that contains basic descriptive statistics of the data.
