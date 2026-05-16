install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

install-deploy:
	pip install -r requirements-deploy.txt

train:
	python -m src.training.train --config configs/train.yaml

evaluate:
	python -m src.evaluation.evaluate --config configs/train.yaml

manual-test:
	python -m src.evaluation.manual_tests --config configs/model.yaml

export:
	python -m src.export.export_model --config configs/train.yaml

run-api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest

lint:
	ruff check .

docker-build:
	docker build -t hate-speech-api .

docker-run:
	docker run -p 8000:8000 hate-speech-api
