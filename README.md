# Feature Engineering

## About

This project is a part of the [Eubucco](https://eubucco.com/) project.
It s a rewite of the feature engineering found at [UFO Map](https://github.com/ai4up/ufo-map/tree/6b9fe3ced499e0859f1b58f710f9436b0da93014/ufo_map/Feature_engineering)


## Development

### Pre-Commit Hooks

To maintain code quality and consistency, this project uses [pre-commit](https://pre-commit.com/) hooks. These hooks automatically format and lint code before commits, using the following tools:
* [Black](https://github.com/psf/black): Enforces consistent code formatting.
* [Flake8](https://github.com/PyCQA/flake8): Checks for PEP 8 compliance and common Python errors.
* [isort](https://github.com/PyCQA/isort): Sorts and organizes imports.
* [pre-commit-hooks](https://github.com/pre-commit/): Smaller out-of-the-box hooks including checks for end-of-file, trailing whitespace, and large files.

1. Install pre-commit dev dependency:
   ```bash
   pip install pre-commit
   ```
1. Install hooks:
    ```bash
    pre-commit install
    ```
1. Run run all hooks manually:
    ```bash
    pre-commit run --all-files
    ```
