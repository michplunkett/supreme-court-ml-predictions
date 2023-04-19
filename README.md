# Supreme Court ML Prediction Project
The project uses historic supreme court cases to train natural language processing models to predict case rulings.

### Project Requirements
- Python version: `^3.11`
- [Poetry](https://python-poetry.org/)

### Technical Notes
- Any modules should be added via the `poetry add [module]` command.
  - Example: `poetry add pytest`

### Instructions to Run the Project
1. Go into the base directory of the repository and type `poetry shell` into the terminal.
2. Use the `make run` command.

## Standard Commands
- `make format`: Runs `Black` on the codebase
- `make lint`: Runs `pytlint` on the codebase
- `make test`: Runs test cases in the `test` directory
- `make run`: Runs the `main` function in the `supreme-court-predictions` folder
- `make get-data`: This function gets the initial data from [Convokit](https://convokit.cornell.edu/documentation/supreme.html)
- `make clean-data`: This function cleans the downloaded [Convokit](https://convokit.cornell.edu/documentation/supreme.html)
