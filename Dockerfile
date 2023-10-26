FROM python:3.9.18-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt
COPY mlp-model/ /app/mlp-model/
COPY *.py /app/
EXPOSE 8080
ENTRYPOINT [ "python3", "app.py" ]