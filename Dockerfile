# LangGraph API for the hike planner.
# Uses `langgraph dev` (in-memory Agent Server). That server speaks the API
# agent-chat-ui needs and does not require a LangSmith or LangGraph license.
# `langgraph up` does, so this image does not use it.
#
# Node is included so the weather MCP server, and the local NPS MCP server
# when NPS_API_KEY is set, can be started with npx. If npx fails, the Python
# agents fall back to Open-Meteo, api.weather.gov, and OpenStreetMap.

FROM node:22-bookworm-slim AS nodebin

FROM python:3.12-slim-bookworm

COPY --from=nodebin /usr/local/bin/node /usr/local/bin/node
COPY --from=nodebin /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && node --version \
    && npx --version

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PROVIDER=openai \
    PORT=2024

COPY requirements.txt pyproject.toml README.md langgraph.json main.py ./
COPY hike_agent ./hike_agent

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps .

EXPOSE 2024

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '2024'), timeout=4)"

CMD ["sh", "-c", "langgraph dev --host 0.0.0.0 --port ${PORT:-2024} --no-browser --no-reload --allow-blocking"]
