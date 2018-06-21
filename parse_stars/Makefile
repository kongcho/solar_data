setup:
	virtualenv --python=python venv
	venv/bin/pip install -r ./requirements.txt

run:
	venv/bin/python ./parse_plot.py

clean:
	rm -rf venv
	rm -f *.pyc
	rm -f *.log
