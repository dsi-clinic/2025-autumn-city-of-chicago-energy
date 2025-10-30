# 2025-autumn-city-of-chicago-energy

## Project Background

The Department of Technology and Innovation (DTI) is the City of Chicago’s central IT
agency. DTI manages the City’s core technology infrastructure, digital services, data
management, data analytics, and applied data science. Our data team applies data science
and advanced analytics, including model development, forecasting, and pattern detection,
to strengthen decision-making, improve service delivery, and build more accessible,
resident-centered digital tools.

This project will evaluate the impact of Chicago’s Energy Rating Placards, introduced in 2019, on building energy performance. Using multi-year benchmarking data (2015–2024), the analysis will examine how energy efficiency, greenhouse gas intensity, and other performance indicators have evolved across time, building types, and neighborhoods.

The study will focus on statistical and predictive modeling to measure changes and identify drivers of performance improvements. Visualization will be used as a supporting tool to communicate results, not as a primary deliverable.

Key questions include:
- Have energy efficiency and greenhouse gas intensity improved since the introduction of placards, citywide and across building categories?
- Do buildings that started with lower energy ratings show greater improvements compared to higher-rated peers?
- Which building characteristics (size, age, use type, energy mix) are most strongly associated with performance changes over time?


## Project Goals

1. Data Integration and Preparation
    - Compile benchmarking data from 2015–2024.
    - Clean and preprocess data: handle missing values, normalize metrics, align variables across years.
    - Construct a dataset suitable for trend analysis and modeling.
2. Exploratory Data Analysis (EDA)
    - Summary statistics for Energy Star scores, EUI, greenhouse gas emissions, and water use before and after the introduction of placards.
    - Visualizations (trend lines, boxplots, heatmaps) to highlight temporal and categorical patterns.
    - Correlation analysis between energy metrics and building characteristics.
3. Comparative and Statistical Analysis
    - Pre/post comparisons (2015–2018 vs. 2019–2024) of energy efficiency and emissions.
    - Subgroup comparisons by building type and initial rating level.
    - Regression-based tests (including difference-in-differences style models) to evaluate shifts associated with placards.
4. Predictive Modeling
    - Develop models (logistic regression, random forest, gradient boosting) to identify building characteristics linked to performance improvement.
    - Evaluate models with cross-validation and performance metrics (ROC-AUC, precision/recall).
    - Use feature importance analysis to identify the strongest predictors.
5. Final Report
    - Synthesize findings from EDA, comparative analysis, and modeling.
    - Include clear visualizations (trend plots, model feature importance charts) to illustrate results.
    - Provide policy-relevant insights on the role of placards in driving building performance improvements.

## Data

- Chicago Energy Benchmarking: Includes property locations and sizes along with their energy metrics such as the energy star score, the total site energy use intensity (EUI), water use, electricity use, natural gas use, and other fuel use.
    - [2014 Data Reported in 2015](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2014-Data-Reported-in-/tepd-j7h5/about_data)
    - [2015 Data Reported in 2016](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2015-Data-Reported-in-/ebtp-548e/about_data)
    - [2016 Data Reported in 2017](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2016-Data-Reported-in-/fpwt-snya/about_data)
    - [2017 Data Reported in 2018](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2017-Data-Reported-in-/j2ev-2azp/about_data)
    - [2018 Data Reported in 2019 (First year with Chicago Energy Rating)](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2018-Data-Reported-in-/m2kv-bmi3/about_data)
    - [2019 Data Reported in 2020](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2019-Data-Reported-in-/jn94-it7m/about_data)
    - [2020 Data Reported in 2021](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2020-Data-Reported-in-/ydbk-8hi6/about_data)
    - [2021 Data Reported in 2023](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2021-Data-Reported-in-/gkf4-txtp/about_data)
    - [2022 Data Reported in 2023](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2022-Data-Reported-in-/mz3g-jagv/about_data)
    - [2023 Data Reported in 2024](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking-2023-Data-Reported-in-/3a36-5x9a/about_data)

## Other Resources
- [Chicago Energy Benchmarking](https://www.chicago.gov/city/en/progs/env/building-energy-benchmarking---transparency.html)
- [Launch of Chicago Energy Rating System](https://www.chicago.gov/city/en/depts/mayor/press_room/press_releases/2019/august/EnergyRatingSystem.html)


## First Week

- Complete the quick start below, making sure that you can find the file `sample_output.csv`.
- Start data integration and preparation (Goal 1)
- Identify key variables of interest and create visualizations to begin exploring patterns


### Docker

## Quick Start

### 1. Setup Environment
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env to set your data directory path
# Example: DATA_DIR=/Users/yourname/project/data
```

### 2. Install Pre-commit Hooks
```bash
make run-interactive
# Inside container:
cd src
pre-commit install
exit
```

### 3. Test Your Setup
```bash
make test-pipeline
```

If successful, you should see `sample_output.csv` appear in your data directory.

## Technical Expectations

### Pre requisites:

We use Docker, Make and uv as part of our curriculum. If you are unfamiliar with them, it is strongly recommended you read over the following:
- [An introduction to Docker](https://docker-curriculum.com/)
- [An introduction to uv](https://realpython.com/python-uv/)

### Container-Based Development

**All code must be run inside the Docker container.** This ensures consistent environments across different machines and eliminates "works on my machine" issues.

### Environment Management with uv

We use [uv](https://docs.astral.sh/uv/) for Python environment and package management _inside the container_. uv handles:
- Virtual environment creation and management (replaces venv/pyenv)
- Package installation and dependency resolution (replaces pip)
- Project dependency management via `pyproject.toml`

**Important**: When running Python code, prefix commands with `uv run` to maintain the proper environment:

```bash
# Example: Running the pipeline
uv run python src/utils/pipeline_example.py

# Example: Running a notebook
uv run jupyter lab

# Example: Running tests
uv run pytest
```

### Container Volume Structure

```
Container: /project/
├── src/           # Your source code (mounted from host repo)
├── data/          # Data directory (mounted from HOST_DATA_DIR)
├── .venv/         # Python virtual environment (created in container)
├── pyproject.toml # Project configuration
└── ...
```


## Usage & Testing

- Set `DATA_DIR` in your `.env` file to specify where data lives on your host
- This directory is mounted to `/project/data` inside the container
- Keep data separate from code to avoid repository bloat and enable easy data sharing

Run the command `make test-pipeline`. If your setup is working you should see a file `sample_output.csv` appear in your data directory. 


### Docker & Make

We use `docker` and `make` to run our code. There are three built-in `make` commands:

* `make build-only`: This will build the image only. It is useful for testing and making changes to the Dockerfile.
* `make run-notebooks`: This will run a Jupyter server, which also mounts the current directory into `/program`.
    * You may use notebooks within VSCode by copying the local URL of the Jupyter Server and selecting it as your kernel. (i.e. `http://localhost:8888/lab?token=abcdefg`)
* `make run-interactive`: This will create a container (with the current directory mounted as `/program`) and load an interactive session. 

The file `Makefile` contains details about the specific commands that are run when calling each `make` target.




## Style
We use [`ruff`](https://docs.astral.sh/ruff/) to enforce style standards and grade code quality. This is an automated code checker that looks for specific issues in the code that need to be fixed to make it readable and consistent with common standards. `ruff` is run before each commit via [`pre-commit`](https://pre-commit.com/). If it fails, the commit will be blocked and the user will be shown what needs to be changed.

Once you have followed the quick setup instructions above for installing dependencies, you can run:
```bash
pre-commit run --all-files
```

You can also run `ruff` directly:
```bash
ruff check
ruff format
```