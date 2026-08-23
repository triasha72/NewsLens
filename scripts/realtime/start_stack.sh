#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
compose_file="$repository_root/deploy/realtime/compose.yaml"

docker compose -f "$compose_file" up --detach --build --wait
"$repository_root/scripts/realtime/health_check.sh"
