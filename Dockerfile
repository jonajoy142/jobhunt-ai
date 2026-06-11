FROM apify/actor-python:3.11

COPY requirements.txt ./
RUN pip install -r requirements.txt --no-cache-dir

COPY my_actor/ ./my_actor/
RUN python -m compileall -q my_actor/

CMD ["python", "-m", "my_actor"]
