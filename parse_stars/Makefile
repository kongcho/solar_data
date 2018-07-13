setup:
	virtualenv --python=python venv
	venv/bin/pip install -r ./requirements.txt

run:
	venv/bin/python ./main.py

model:
	venv/bin/python ./model.py

clean:
	rm -rf venv
	rm -f *.pyc
	rm -f *.log
