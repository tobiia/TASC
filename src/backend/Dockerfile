# ubuntu
FROM python:3.12-slim

# ensures python stdout/err is sent to terminal in real time
# b/c skips buffering
ENV PYTHONUNBUFFERED=1

# creating arbitrary directory inside container file sys
WORKDIR /app

# COPY copies the contents of 1st arg, not the arg itself
# so dest needs to declare directs if you want

# copy dependency files first
COPY requirements.txt .
COPY config.toml .

COPY packages ./packages

# create venv
RUN python -m venv .venv

# using venv python/pip from now on
ENV PATH="/app/.venv/bin:$PATH"

RUN python -m pip install --upgrade pip

# install python dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN python -m pip install ./packages/LSC ./packages/Top2Vec

# install spaCy model
RUN python -m spacy download en_core_web_sm

# app code + packages
COPY src ./src
COPY corpora ./corpora

ENV PYTHONPATH=/app

CMD [ "python", "-m", "uvicorn", "src.backend.app.main:app", "--host", "0.0.0.0", "--port", ${PORT_BACKEND} ]