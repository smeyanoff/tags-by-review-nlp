initial_data_path := "./data/initial_data"
intermediate_data_path := "./data/intermediate_data"
prepared_data_path := "./data/prepared_data"
creds_path := "./creds"

start: install_poetry install_deps

install_poetry:
	python3 -m pip install poetry

install_deps: install_poetry
	poetry install

make_dirs:
	mkdir -p $(initial_data_path)
	mkdir -p $(intermediate_data_path)
	mkdir -p $(prepared_data_path)
	mkdir -p $(creds_path)

push: install_deps make_dirs
	poetry run dvc add data
	poetry run dvc add creds
	poetry run dvc add tables  
	poetry run dvc add models/weights
	poetry run dvc push

pull: install_deps make_dirs
	poetry run dvc pull
