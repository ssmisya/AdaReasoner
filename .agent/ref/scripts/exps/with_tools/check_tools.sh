#!/usr/bin/env bash
set -euo pipefail

CTRL="${CONTROLLER_ADDR:-http://127.0.0.1:21112}"
QUIET=0
[[ "${1:-}" == "--quiet" ]] && QUIET=1

log() { (( QUIET )) || echo "$*"; }
fail() { (( QUIET )) || echo "ERROR: $*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || fail "curl is not installed"
command -v python3 >/dev/null 2>&1 || fail "python3 is not installed"
command -v ss >/dev/null 2>&1 || fail "ss is not installed"

response="$(curl -fsS -m 5 -X POST "$CTRL/list_models")" || fail "tool controller is unavailable: $CTRL"
python3 - "$response" <<'PY' || fail "controller has not registered Point, OCR and Crop"
import json, sys
registered = set(json.loads(sys.argv[1]).get("models", []))
required = {"Point", "OCR", "Crop"}
missing = sorted(required - registered)
if missing:
    raise SystemExit("missing: " + ", ".join(missing))
PY
log "OK controller registry: Point, OCR, Crop"

for spec in \
    "21112:Controller" \
    "50002:Point-GPU1" \
    "50003:Point-GPU2" \
    "50010:OCR-GPU1" \
    "50011:OCR-GPU2" \
    "50012:Crop-CPU"; do
    port="${spec%%:*}"
    label="${spec#*:}"
    ss -tln 2>/dev/null | grep -qE "[:.]${port}[[:space:]]" || fail "$label is not listening on port $port"
    if [[ "$port" != "21112" ]]; then
        curl -fsS -m 10 -X POST "http://127.0.0.1:${port}/worker_get_status" >/dev/null \
            || fail "$label health endpoint failed on port $port"
    fi
    log "OK $label :$port"
done

log "Tool services are ready."
