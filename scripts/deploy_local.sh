#!/usr/bin/env bash
# Deploy the serving API to local Docker Compose. Never remove Docker volumes.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

COMPOSE=(sudo docker compose -f compose.yaml -f compose.serving.yaml)

if curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  # Don't collide with an already-running native Uvicorn server.
  if ! "${COMPOSE[@]}" ps --services --status running 2>/dev/null | grep -qx serving; then
    echo 'Port 8000 is already serving another process (possibly native Uvicorn).'
    echo 'Stop that server in its terminal before Docker deployment.'
    exit 2
  fi
fi

"${COMPOSE[@]}" config --quiet
"${COMPOSE[@]}" up -d --build serving

healthy=false
for attempt in $(seq 1 24); do
  if curl -fsS --max-time 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("status")=="ok" and d.get("loaded_models")==2' >/dev/null 2>&1; then
    healthy=true
    break
  fi
  sleep 5
done

if [[ "$healthy" != true ]]; then
  echo 'Serving failed health checks; inspect logs:' >&2
  echo 'sudo docker compose -f compose.yaml -f compose.serving.yaml logs --tail=100 serving' >&2
  exit 1
fi

curl -fsS http://127.0.0.1:8000/models | python3 -m json.tool
printf '\nPASS: Serving API is healthy with two loaded production models.\n'
