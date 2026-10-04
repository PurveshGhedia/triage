#!/usr/bin/env bash
# Deploy TRIAGE to Cloud Run: three PRIVATE services built from one image.
#
#   triage-agent1     Agent 1 (perception) as an A2A service       runs as its own service account
#   triage-agent2     Agent 2 (planner)    as an A2A service       runs as its own service account
#   triage-dashboard  the UI and the orchestrator, calls both agents with ID tokens
#
# Nothing is public (this project has a domain-restricted-sharing policy). You open the dashboard with
#   gcloud run services proxy triage-dashboard --region us-central1 --port 8080
#
#   bash deploy/deploy.sh                    # everything
#   DRY_RUN=1 bash deploy/deploy.sh          # print the commands, change nothing
#   STEP=build bash deploy/deploy.sh         # one step only (see STEPS below)
#
# Settings (environment): PROJECT REGION REPO VISION_VARIANT TRIAGE_STORE ALLOW_RUN
STEPS="apis repo accounts iam stage build agents dashboard bindings"
set -eu

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-triage}"
VISION_VARIANT="${VISION_VARIANT:-classes}"
TRIAGE_STORE="${TRIAGE_STORE:-firestore}"
ALLOW_RUN="${ALLOW_RUN:-1}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${STEP:-}"

[ -n "$PROJECT" ] || { echo "No project: set PROJECT or run 'gcloud config set project ...'"; exit 1; }

run() {                       # print always; execute unless DRY_RUN
  echo "+ $*"
  [ "$DRY_RUN" = "1" ] && return 0
  "$@"
}
step() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
heading() { echo; echo "== $*"; }

NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)' 2>/dev/null || true)"
[ -n "$NUMBER" ] || { [ "$DRY_RUN" = "1" ] && NUMBER="000000000000" || { echo "Could not read the project number"; exit 1; }; }
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/triage:latest"
SA1="triage-agent1@$PROJECT.iam.gserviceaccount.com"
SA2="triage-agent2@$PROJECT.iam.gserviceaccount.com"
SAD="triage-dashboard@$PROJECT.iam.gserviceaccount.com"
URL1="https://triage-agent1-$NUMBER.$REGION.run.app"     # Cloud Run's URLs are predictable, so the agent
URL2="https://triage-agent2-$NUMBER.$REGION.run.app"     # card can advertise its own address from the start
COMMON_ENV="GCP_PROJECT_ID=$PROJECT,VERTEX_AI_LOCATION=$REGION,USE_MOCK_TOOLS=true,TRIAGE_VISION_CACHE=/app/data/cache/vision"

echo "TRIAGE deploy  project=$PROJECT ($NUMBER)  region=$REGION  dry-run=$DRY_RUN${ONLY:+  only=$ONLY}"

if step apis; then
  heading "Enable APIs"
  run gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
      firestore.googleapis.com aiplatform.googleapis.com --project "$PROJECT"
fi

if step repo; then
  heading "Artifact Registry repository"
  if gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
    echo "  $REPO already exists"
  else
    run gcloud artifacts repositories create "$REPO" --repository-format docker --location "$REGION" --project "$PROJECT"
  fi
fi

if step accounts; then
  heading "Service accounts (one per service, so each gets only what it needs)"
  for name in triage-agent1 triage-agent2 triage-dashboard; do
    if gcloud iam service-accounts describe "$name@$PROJECT.iam.gserviceaccount.com" --project "$PROJECT" >/dev/null 2>&1; then
      echo "  $name already exists"
    else
      run gcloud iam service-accounts create "$name" --display-name "TRIAGE $name" --project "$PROJECT"
    fi
  done
fi

if step iam; then
  heading "Project roles (least privilege)"
  # Agent 1 only calls Gemini. Agent 2 needs no Google API at all. The dashboard stores tiles in
  # Firestore and may call Gemini itself (fallback / tool selection).
  run gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SA1" --role roles/aiplatform.user --condition=None --quiet
  run gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SAD" --role roles/datastore.user --condition=None --quiet
  run gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$SAD" --role roles/aiplatform.user --condition=None --quiet
fi

if step stage; then
  heading "Stage the build context (code + demo data + cached vision answers only)"
  run python -m deploy.stage_context
fi

if step build; then
  heading "Build the image with Cloud Build (no local Docker needed)"
  run gcloud builds submit deploy/_context --tag "$IMAGE" --project "$PROJECT"
fi

deploy_agent() {              # name  service-account  public-url  agent-kind
  run gcloud run deploy "$1" --image "$IMAGE" --project "$PROJECT" --region "$REGION" \
      --service-account "$2" --no-allow-unauthenticated \
      --memory 1Gi --cpu 1 --timeout 900 --max-instances 2 \
      --command python --args="-m,scripts.serve_agents,--agent,$4,--host,0.0.0.0" \
      --set-env-vars "$COMMON_ENV,TRIAGE_PUBLIC_URL=$3" --quiet
}

if step agents; then
  heading "Deploy the agent services (private)"
  deploy_agent triage-agent1 "$SA1" "$URL1" perception
  deploy_agent triage-agent2 "$SA2" "$URL2" planner
fi

if step dashboard; then
  heading "Deploy the dashboard (private; single instance so run state stays consistent)"
  run gcloud run deploy triage-dashboard --image "$IMAGE" --project "$PROJECT" --region "$REGION" \
      --service-account "$SAD" --no-allow-unauthenticated \
      --memory 1Gi --cpu 1 --timeout 900 --max-instances 1 \
      --set-env-vars "$COMMON_ENV,TRIAGE_AGENT1_URL=$URL1,TRIAGE_AGENT2_URL=$URL2,TRIAGE_A2A_AUTH=id_token,TRIAGE_STORE=$TRIAGE_STORE,TRIAGE_ALLOW_RUN=$ALLOW_RUN,TRIAGE_VISION_VARIANT=$VISION_VARIANT" --quiet
fi

if step bindings; then
  heading "Let the dashboard (and only the dashboard) call the agents"
  for svc in triage-agent1 triage-agent2; do
    run gcloud run services add-iam-policy-binding "$svc" --project "$PROJECT" --region "$REGION" \
        --member "serviceAccount:$SAD" --role roles/run.invoker --quiet
  done
  for pair in "triage-agent1:$URL1" "triage-agent2:$URL2"; do
    svc="${pair%%:*}"; expected="${pair#*:}"
    actual="$(gcloud run services describe "$svc" --project "$PROJECT" --region "$REGION" --format='value(status.url)' 2>/dev/null || true)"
    if [ "$DRY_RUN" != "1" ] && [ -n "$actual" ] && [ "$actual" != "$expected" ]; then
      echo "  !! $svc is at $actual, not $expected: its agent card advertises the wrong address."
      echo "     Fix: re-run with the real URL, e.g.  gcloud run services update $svc --region $REGION --update-env-vars TRIAGE_PUBLIC_URL=$actual"
    fi
  done
fi

echo
echo "Done. Open the dashboard (nothing here is public):"
echo "  gcloud run services proxy triage-dashboard --region $REGION --project $PROJECT --port 8080"
echo "  then visit http://localhost:8080"
echo "Check it end to end with:  bash deploy/smoke_test.sh"
