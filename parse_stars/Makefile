setup:
	virtualenv --python=python2 venv
	venv/bin/pip2 install -r ./requirements.txt

run:
	venv/bin/python ./main.py

model:
	venv/bin/python ./model.py

quick:
	rm -f *.pyc
	rm -f *.log
	rm -f screenlog.0
	rm -f *.tar.gz
	rm -f *.zip
	rm -f *.out
	rm -f f3/f3/*pyc
	rm -f f3/f3/__pycache__/

clean:
	rm -rf venv
	rm -f *.pyc
	rm -f *.log
