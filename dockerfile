FROM python:3.9.1
ADD . /python-flask
WORKDIR /python-flask
RUN pip install -r requirements.txt

CMD ["env","FLASK_APP=main.py","flask","run","--host=0.0.0.0"]