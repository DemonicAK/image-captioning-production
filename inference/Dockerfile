FROM tensorflow/tensorflow:2.17.0


ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /inference

# System deps
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir uv

# Copy dependency files first (cache-friendly)
COPY inference/pyproject.toml inference/uv.lock ./
# COPY inference/ ./

# Install deps into system Python
RUN uv pip install --system --no-cache --requirements pyproject.toml


# Copy app code
COPY . .

EXPOSE 8000

CMD ["bash", "scripts/entrypoint.sh"]