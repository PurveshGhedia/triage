#!/usr/bin/env bash
# End-to-end check of the deployed (private) services, using your own identity token.
#   bash deploy/smoke_test.sh [project] [region]
set -u
PROJECT="${1:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${2:-us-central1}"
DISASTER="${DISASTER:-hurricane-harvey}"
fails=0
ok()  { echo "  [ OK ] $*"; }
bad() { echo "  [FAIL] $*"; fails=$((fails + 1)); }
url_of() { gcloud run services describe "$1" --project "$PROJECT" --region "$REGION" --format='value(status.url)' 2>/dev/null; }

DASH="$(url_of triage-dashboard)"; A1="$(url_of triage-agent1)"; A2="$(url_of triage-agent2)"
TOKEN="$(gcloud auth print-identity-token 2>/dev/null)"
[ -n "$DASH" ] && [ -n "$A1" ] && [ -n "$A2" ] && ok "found all three services" || { bad "a service is missing (run deploy first)"; exit 1; }
auth=(-H "Authorization: Bearer $TOKEN")

echo "1. Private: no credentials must be refused"
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "$DASH/api/health")"
[ "$code" = "403" ] || [ "$code" = "401" ] && ok "dashboard refuses anonymous callers (HTTP $code)" || bad "dashboard answered HTTP $code without credentials"

echo "2. Dashboard with your token"
health="$(curl -s --max-time 60 "${auth[@]}" "$DASH/api/health")"
echo "$health" | grep -q '"ok":true' && ok "health ok: $health" || bad "health failed: $health"
echo "$health" | grep -q '"agents":"remote"' && ok "dashboard is wired to the remote agents" || bad "dashboard is not using remote agents"

echo "3. Agent cards"
for pair in "perception_agent:$A1" "planner_agent:$A2"; do
  name="${pair%%:*}"; url="${pair#*:}"
  curl -s --max-time 60 "${auth[@]}" "$url/.well-known/agent-card.json" | grep -q "$name" && ok "$name card served" || bad "$name card not served"
done

echo "4. A full run through the agents over A2A (mock tools, no live feed)"
run_id="$(curl -s --max-time 60 "${auth[@]}" -H 'Content-Type: application/json' \
  -d "{\"disaster\": \"$DISASTER\", \"transport\": \"a2a\", \"wave_size\": null, \"tools\": \"mock\"}" "$DASH/api/runs" \
  | python3 -c 'import sys, json; print(json.load(sys.stdin).get("run_id", ""))' 2>/dev/null)"
if [ -z "$run_id" ]; then bad "could not start a run"; else
  ok "started run $run_id"
  for _ in $(seq 1 60); do
    sleep 3
    ev="$(curl -s --max-time 60 "${auth[@]}" "$DASH/api/runs/$run_id/events")"
    echo "$ev" | grep -q '"finished":true' && break
  done
  echo "$ev" | python3 -c '
import sys, json
d = json.load(sys.stdin); e = d["events"]
types = [x["type"] for x in e]
fb = [x for x in e if x["type"] == "transport_fallback"]
fin = [x for x in e if x["type"] == "run_finished"]
print("  events:", len(e), "| fallbacks:", len(fb), "| finished:", bool(fin))
sys.exit(0 if fin and not fb else 1)' && ok "run finished over A2A with no fallback" || bad "run did not finish cleanly over A2A (a fallback means an agent call failed)"
fi
echo
[ "$fails" = "0" ] && echo "SMOKE TEST PASSED" || echo "SMOKE TEST: $fails check(s) failed"
exit $((fails > 0))
