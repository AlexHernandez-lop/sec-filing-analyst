.PHONY: install test ingest run chat clean

install:
	pip install -r requirements.txt
	pip install pytest

test:
	pytest tests/ -v

ingest:
	python cli.py ingest $(COMPANY) --forms $(or $(FORMS),10-K) --limit $(or $(LIMIT),2)

run:
	streamlit run app.py

chat:
	python cli.py chat

list:
	python cli.py list

clean:
	rm -rf data/
	find . -type d -name __pycache__ -exec rm -rf {} +
