# AksharaHarness — web UI in a container.
#
# Build:  podman build -t localhost/akshara-web .
# Run:    podman run -d --name akshara-web -p 8400:8321 \
#             -v ./.env:/app/.env:ro localhost/akshara-web
#         (see README "Run in a container")
#
# The image carries code only — no keys. Secrets arrive at run time,
# either mounted as a read-only .env or as plain environment variables.

FROM python:3.12-slim

# The project manages its Python with uv; bring it in without a second
# image pull, then install dependencies before the source so code edits
# don't re-resolve the whole lockfile.
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra web

COPY src/ src/
RUN uv sync --frozen --extra web

ENV PATH="/app/.venv/bin:$PATH"

# A dedicated user: the agent's tools run inside this filesystem, so
# they shouldn't be root in it either.
RUN useradd -m akshara && chown -R akshara:akshara /app
USER akshara

EXPOSE 8321

# ENTRYPOINT, not CMD-with-everything: extra `podman run` args then
# COMPOSE instead of replacing ("image --web --cwd /workspace" works).
# Bind 0.0.0.0 — the default loopback bind is unreachable from outside
# the container no matter how you publish the port.
ENTRYPOINT ["akshara"]
CMD ["--web", "--host", "0.0.0.0"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as u,sys; \
sys.exit(0 if u.urlopen('http://127.0.0.1:8321/', timeout=2).status == 200 else 1)"
