###
start:
	install_poetry
	install_dependences
	install_spacy_model
	install_pre_commit

install_dependences:
	poetry install

install_poetry:
	python -m pip install poetry

install_spacy_model:
	poetry run python -m spacy download ru_core_news_md

install_pre_commit:
	poetry run pre-commit install
	poetry run pre-commit autoupdate
