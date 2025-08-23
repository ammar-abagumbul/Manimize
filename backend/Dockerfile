FROM python:3.13-slim AS base

FROM base AS builder

RUN apt-get update && \
    apt-get install -y gcc g++ gfortran musl-dev texlive-latex-base libx11-dev pkg-config cmake libcairo2-dev libpango1.0-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BITECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/usr/src/app/.venv

WORKDIR /usr/src/app

COPY uv.lock pyproject.toml /usr/src/app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY . /usr/src/app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM base
COPY --from=builder /usr/src/app /usr/src/app
ENV PATH="/usr/src/app/.venv/bin:$PATH"
EXPOSE 8000

CMD ["fastapi", "dev", "main.py", "--host", "0.0.0.0", "--port", "8000"]
