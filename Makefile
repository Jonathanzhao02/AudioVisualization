PYTHON3=${CONDA_PREFIX}/bin/python3

init: environment.yml
	conda env create -f environment.yml

run:
	$(PYTHON3) main.py

update: environment.yml
	conda env update --name audio_py --file environment.yml --prune

export:
	conda env export --name audio_py --no-builds | grep -v "^prefix: " > environment.yml
