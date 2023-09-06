# HIIT-vs-MICT Project

## Quick Start
See the [Setup Guide](docs/setup_guide.md) to get started analyzing exercise data!

## Repo Structure
```
├── .gitignore
├── LICENSE
├── README.md
├── config
│   └── .env
├── docs
│   ├── README.md
│   ├── analysis_overview.md
│   ├── code_documentation
│   └── setup_guide.md
├── models
│   ├── README.md
│   └── experiments
└── src
    ├── py
    │   ├── lib_ml
    │   ├── lib_timeseries
    │   ├── notebooks
    │   ├── scripts
    │   └── tests
    └── r
        ├── lib_timeseries
        ├── notebooks
        ├── scripts
        └── tests
```
## Contribution Guidelines
Please follow the below steps to ensure the integrity and quality of our codebase while contributing to this project.

### Branching:
We take a hybrid approach.
<!-- - Always create a new branch for your work. Branch name should reflect the feature or bugfix you're working on. -->
- Always create a new branch for your work.
- Use a **researcher branch** for general research and transition to an appropriate **pipeline branch** or **analysis branch** when contributing code to a particular pipeline process or analysis, respectively.
- Branch names should be descriptive. Please follow the conventions below:

| Branch      | Naming Convention                 | Example                            |
|-------------|-----------------------------------|----------------------------------|
| main        | `'main'`                            |                                  |
| integration | `'integration'`                     |                                  |
| researcher  | `'researcher/[SUNet ID]'`           | `researcher/mcginity`            |
| pipeline    | `'pipeline/[process-description]'`  | `pipeline/polar-data-processing` |
| analysis    | `'analysis/[analysis-description]'` | `analysis/vo2max-estimate`       |


### Commit Messages:
- Write clear, concise commit messages that detail what changes were made and why.

### Testing:
- Add appropriate tests for any new functionality in the `tests` subdirectories.
- Ensure that all relevant tests pass before submitting a pull request.

### Documentation:
- For any new setup instructions or dependencies, update `setup_guide.md`.
- As you create and work on new analyses, update the `analysis_overview.md` accordingly.
- If you introduce a new function, module, or utility, please ensure it's appropriately documented with Python docstrings.
```
def function_name(param1, param2):
    """
    Brief description of the function.

    Args:
        param1 (Type): Description of param1.
        param2 (Type): Description of param2.

    Returns:
        Type: Description of return value.
    """
    pass
```

### Libraries and Utilities:
- Before adding new libraries or utilities, check if a similar functionality exists in the `lib_*` directories.
- Add packages you need to appropriate directories. If required, create a new folder following the `lib_[description]` naming convention and add your packages there.


### Notebooks:
- Ensure that the notebooks are cleanly executed from start to finish.
- Clear all cell outputs before committing to keep the repo size down.
- Add descriptive comments and markdown cells to explain each step.
- Refactor code whenever possible to avoid redundancy.

### Pull Requests (PRs):
- Ensure your PR has a descriptive title and provides a detailed description of the changes made.
- Link any relevant issues.
<!-- - Request code review from at least one other team member. -->
<!-- - Ensure all CI checks pass (if any are set up). -->

### Data Security and Configs:
- Do not commit secrets or sensitive information. Use `.env` for environment-specific settings and ensure it's ignored in `.gitignore`.
- Update the `config` directory for any new configuration settings.
- **Never commit any data to GitHub!**

### Dependencies:
- If you introduce new libraries or tools, update the environment or package files accordingly.
- Ensure compatibility with existing dependencies.

Thank you for your contributions and dedication to maintaining the quality and integrity of this research project! 

## License
This project is licensed under the terms mentioned in [LICENSE](LICENSE) file.
