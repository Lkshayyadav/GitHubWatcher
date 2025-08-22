.PHONY: run format lint freeze

run:
	. venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 5000

format:
	. venv/bin/activate && python -m black . && python -m isort .

lint:
	. venv/bin/activate && python -m black --check . && python -m isort --check-only .

freeze:
	. venv/bin/activate && pip freeze > requirements.txt
