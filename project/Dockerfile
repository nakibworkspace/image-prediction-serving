# pull official base image
FROM python:3.11-slim-bookworm

# set working directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
  && apt-get -y install netcat-traditional gcc g++ postgresql \
  && apt-get -y install libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
  && apt-get -y install libhdf5-dev \
  && apt-get clean

# install python dependencies
RUN pip install --upgrade pip setuptools wheel

# Install main requirements first
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install dev requirements
COPY ./requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# add app
COPY . .

# create uploads directory
RUN mkdir -p /usr/src/app/uploads

# add entrypoint.sh
COPY ./entrypoint.sh .
RUN chmod +x /usr/src/app/entrypoint.sh

# run entrypoint.sh
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]