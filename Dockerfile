FROM tensorflow/tensorflow:latest

WORKDIR .
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENTRYPOINT [ "python" ]

CMD [ "model_server.py" ]