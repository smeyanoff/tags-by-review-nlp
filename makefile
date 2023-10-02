###
start:
	install_poetry
	install_spacy_model
	install_pre_commit

install_poetry:
	python -m pip install poetry

install_spacy_model:
	poetry run python -m spacy download ru_core_news_md

install_pre_commit:
	poetry run pre-commit install
