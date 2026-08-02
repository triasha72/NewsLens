# NewsLens container deployment

NewsLens provides a Dockerized FastAPI recommendation service.

## Prerequisites

- Docker Desktop
- a locally generated NewsLens model artifact
- port 8000 available on the host

The licensed MIND data and generated model artifacts are excluded from Git and
from the Docker image.

## Build the image

```bash
docker build --tag newslens-api:0.2.0 .