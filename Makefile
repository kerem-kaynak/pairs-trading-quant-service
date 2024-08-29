.PHONY: run build test deploy setup run-local pytest print-env

include .env
export $(shell sed 's/=.*//' .env)

run:
	flask run

build:
	docker build -t myapp .

test:
	python -m unittest discover tests

pytest:
	PYTHONPATH=$(PWD) pytest tests/

deploy:
	gcloud run deploy myapp --source .

setup:
	pip install -r requirements.txt

run-local:
	flask run --host=0.0.0.0 --port=8000

print-env:
	@echo "Environment variables:"
	@echo "INSTANCE_CONNECTION_NAME: $(INSTANCE_CONNECTION_NAME)"
	@echo "DB_USER: $(DB_USER)"
	@echo "DB_PASS: $(DB_PASS)"
	@echo "DB_NAME: $(DB_NAME)"
	@echo "API_TOKEN: $(API_TOKEN)"