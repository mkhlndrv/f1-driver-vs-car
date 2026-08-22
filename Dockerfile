FROM python:3.12-slim

# LightGBM links against the OpenMP runtime, which the slim image does not carry. Without it the
# import fails at collection time, so this is the difference between the container running the
# pipeline and not importing it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

COPY . .
RUN uv sync --locked

CMD ["uv", "run", "python", "-m", "f1_skill"]
