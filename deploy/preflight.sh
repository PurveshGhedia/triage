#!/usr/bin/env bash
# Can this GCP project host the TRIAGE demo on Cloud Run?
#
#   bash deploy/preflight.sh                       # uses your gcloud project, us-central1
#   bash deploy/preflight.sh my-project us-central1
#
# Checks account, billing, the APIs we need, and any organisation policy that restricts who can be
# granted access. Then it deploys a throwaway "hello" service and tests whether it is reachable
# WITHOUT credentials (public) and WITH your identity token (private). The throwaway service is
# deleted at the end, even if something fails. Nothing else is created or changed.
set -u

PROJECT="${1:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${2:-us-central1}"
SVC="triage-preflight-hello"
IMAGE="us-docker.pkg.dev/cloudrun/container/hello"
APIS="run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com firestore.googleapis.com aiplatform.googleapis.com"

passed=0; failed=0; missing_apis=""; created=0
ok()   { echo "  [ OK ] $*"; passed=$((passed + 1)); }
bad()  { echo "  [FAIL] $*"; failed=$((failed + 1)); }
note() { echo "         $*"; }

cleanup() {
  if [ "$created" = "1" ]; then
    gcloud run services delete "$SVC" --project "$PROJECT" --region "$REGION" --quiet >/dev/null 2>&1 \
      && echo "  (throwaway service $SVC deleted)" \
      || echo "  !! could not delete $SVC; remove it with: gcloud run services delete $SVC --region $REGION"
  fi
}
trap cleanup EXIT

echo "TRIAGE deployment preflight"
echo "  project: ${PROJECT:-<none>}   region: $REGION"
echo

if ! command -v gcloud >/dev/null 2>&1; then
  bad "gcloud is not installed or not on PATH"; echo; echo "RESULT: cannot continue"; exit 1
fi
if [ -z "${PROJECT:-}" ]; then
  bad "no project set (run: gcloud config set project YOUR_PROJECT_ID)"; echo; echo "RESULT: cannot continue"; exit 1
fi

echo "1. Account and billing"
account="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)"
[ -n "$account" ] && ok "signed in as $account" || bad "not signed in (run: gcloud auth login)"
billing="$(gcloud billing projects describe "$PROJECT" --format='value(billingEnabled)' 2>/dev/null)"
case "$billing" in
  [Tt]rue) ok "billing is enabled" ;;
  *)       bad "billing does not look enabled (got: '${billing:-nothing}')" ;;
esac

echo "2. APIs"
for api in $APIS; do
  found="$(gcloud services list --enabled --project "$PROJECT" --filter="config.name=$api" --format='value(config.name)' 2>/dev/null)"
  if [ -n "$found" ]; then ok "$api enabled"; else bad "$api NOT enabled"; missing_apis="$missing_apis $api"; fi
done
[ -n "$missing_apis" ] && note "enable with: gcloud services enable$missing_apis --project $PROJECT"

echo "3. Organisation policy on who can be granted access"
policy="$(gcloud resource-manager org-policies describe constraints/iam.allowedPolicyMemberDomains --project "$PROJECT" --effective 2>&1)"
if echo "$policy" | grep -qi "allowedValues"; then
  bad "a domain-restricted-sharing policy is active, so 'allUsers' (public access) will probably be refused"
  note "this is the likely cause of the earlier BigQuery 'permitted customer' error"
else
  ok "no domain-restriction policy found (or it is not visible to you)"
fi

echo "4. Reachability test (throwaway Cloud Run service)"
public_state="untested"; private_state="untested"
if echo "$missing_apis" | grep -q "run.googleapis.com"; then
  note "skipped: the Cloud Run API is not enabled yet"
else
  if gcloud run deploy "$SVC" --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
       --allow-unauthenticated --quiet >/tmp/triage_preflight_deploy.log 2>&1; then
    created=1; ok "deployed $SVC"
  else
    # the service may exist even if the public-access step failed
    gcloud run services describe "$SVC" --project "$PROJECT" --region "$REGION" >/dev/null 2>&1 && created=1
    bad "deploy reported errors (see /tmp/triage_preflight_deploy.log)"
    tail -3 /tmp/triage_preflight_deploy.log | sed 's/^/         /'
  fi
  url="$(gcloud run services describe "$SVC" --project "$PROJECT" --region "$REGION" --format='value(status.url)' 2>/dev/null)"
  if [ -n "$url" ]; then
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$url")"
    if [ "$code" = "200" ]; then public_state="works"; ok "reachable with NO credentials (public access works)"
    else public_state="blocked"; bad "no-credentials request got HTTP $code (public access is blocked)"; fi
    token="$(gcloud auth print-identity-token 2>/dev/null)"
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -H "Authorization: Bearer $token" "$url")"
    if [ "$code" = "200" ]; then private_state="works"; ok "reachable WITH your identity token (private access works)"
    else private_state="blocked"; bad "identity-token request got HTTP $code"; fi
  else
    bad "could not read the service URL"
  fi
fi

echo
echo "PASSED: $passed   FAILED: $failed"
echo
echo "VERDICT"
if [ -n "$missing_apis" ]; then
  echo "  Enable the missing APIs first (command above), then run this again."
elif [ "$public_state" = "works" ]; then
  echo "  PUBLIC ACCESS works: the dashboard can have an open URL, and the agent services can stay private."
elif [ "$public_state" = "blocked" ] && [ "$private_state" = "works" ]; then
  echo "  PUBLIC ACCESS blocked, private works: deploy everything private and open the dashboard with"
  echo "    gcloud run services proxy triage-dashboard --region $REGION --port 8080"
  echo "  (still a real Cloud Run deployment; the page is served from GCP through your credentials)."
else
  echo "  Inconclusive: paste this whole output so the cause can be worked out."
fi
[ "$failed" = "0" ] && exit 0 || exit 1
