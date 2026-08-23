#!/usr/bin/env bash
set -euo pipefail

check_json_endpoint() {
  local label="$1"
  local url="$2"
  local attempts=30

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if curl --fail --silent --show-error "$url" >/dev/null; then
      echo "$label is ready: $url"
      return 0
    fi
    sleep 2
  done

  echo "$label did not become ready: $url" >&2
  return 1
}

check_json_endpoint "ingestion API" "${NEWSLENS_INGESTION_URL:-http://127.0.0.1:8080}/ready"
check_json_endpoint "consumer 1" "${NEWSLENS_CONSUMER_1_URL:-http://127.0.0.1:8081}/ready"
check_json_endpoint "consumer 2" "${NEWSLENS_CONSUMER_2_URL:-http://127.0.0.1:8082}/ready"
check_json_endpoint "search API" "${NEWSLENS_SEARCH_URL:-http://127.0.0.1:8000}/realtime/ready"
