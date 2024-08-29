# Web Application to Visualize the Performance of Machine Learning Aided Pairs Trading Strategies

![Archutecture Diagram](https://i.ibb.co/F4GWF1g/Pairs-Trading-Architecture.png)

This repository is part of a project that aims to showcase the performance of a machine learning assisted pairs trading strategy developed for a thesis. The project consists of 4 separate services in different repositories.

- [Orchestrator](https://github.com/kerem-kaynak/pairs-trading-orchestrator): Orchestrator of data pipelines and processing workflows. Runs scheduled ETL jobs.
- [Quant / ML Service](https://github.com/kerem-kaynak/pairs-trading-quant-service): Web server exposing endpoints to perform machine learning tasks.
- [Backend API](https://github.com/kerem-kaynak/pairs-trading-backend): Backend API serving data to the client.
- [Frontend](https://github.com/kerem-kaynak/pairs-trading-frontend): Frontend application for web access.

The research in the thesis leading to this project can be found [here](https://github.com/kerem-kaynak/pairs-trading-with-ml) with deeper explanations of the financial and statistical concepts.

## Pairs Trading Quant Service

This service handles all machine learning applications and statistical calculations for trading operations. Whenever the backend or the orchestrator require such workflows, they will interface with this service. This service does not interact with the database directly but through the orchestrator and the backend.

# Technologies

The service is built using Python Flask. It relies heavily on a multitude of libraries for machine learning and statistical analysis such as SciPy, Sci-kit Learn, NumPy, Pandas, etc. It also consists of a comprehensive test suite written in pytest.

# Project Structure

The service is pretty simple with only a handful of endpoints. The route level of the project has:

- `routes`: Route definitions for the endpoints
- `schemas`: JSON schemas for request validation
- `tests`: Unit and integration tests using pytest
- `utils`: Most of the business logic in the routes are abstracted to util functions, this directory is where most of the machine learning and statistical applications are defined

The project also has a Dockerfile and a Makefile for local development.

# Requirements

- Python 3.10+
- Docker / Docker Compose
- Make

# Local Development

Install dependencies:
```
make setup
```

Create & populate a .env file:
```
API_TOKEN=
```

Run locally:
```
make run-local
```

Optionally, run test suite:
```
make pytest
```
