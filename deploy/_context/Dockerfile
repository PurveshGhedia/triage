FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Fail the BUILD, not the demo, if anything the services import is missing.
RUN GCP_PROJECT_ID=build-check VERTEX_AI_LOCATION=us-central1 python -c "\
import dashboard.server, scripts.serve_agents, scripts.run_e2e, shared.firestore_client, \
agents.perception.llm_selector, tools.vision.client; print('imports ok')"

ENV PORT=8080 \
    USE_MOCK_TOOLS=true \
    TRIAGE_VISION_CACHE=/app/data/cache/vision
EXPOSE 8080

# The dashboard by default. The agent services override this with --command/--args.
CMD ["python", "-m", "dashboard.server", "--host", "0.0.0.0"]
