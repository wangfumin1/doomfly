#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV="${SNN_VENV:-$ROOT/.venv-neural}"
RUNS="$ROOT/outputs/doom-learning/research-runs"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

log() { printf '\n[snn-research] %s\n' "$*"; }
die() { printf '\n[snn-research] ERROR: %s\n' "$*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

activate_env() {
  [[ -x "$VENV/bin/python" ]] || die "Python environment missing. Run: bash scripts/snn_research_local.sh bootstrap"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
}

bootstrap() {
  require_cmd "$PYTHON_BIN"
  require_cmd clang++
  local version
  version="$($PYTHON_BIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  [[ "$version" == "3.11" ]] || die "Python 3.11 required; found $version via $PYTHON_BIN"

  if [[ ! -x "$VENV/bin/python" ]]; then
    log "Creating Python 3.11 environment at $VENV"
    "$PYTHON_BIN" -m venv "$VENV"
  fi
  activate_env
  python -m pip install --upgrade pip
  python -m pip install \
    -r requirements-neural.txt \
    -r doom/requirements.txt \
    --build-constraint neural-build-constraints.txt
  log "Environment ready"
}

download_and_verify_data() {
  python - <<'PY'
from pathlib import Path
import hashlib
import json
import urllib.request

name = 'malecns_v1'
registry = json.loads(Path('doom/datasets.json').read_text())['datasets'][name]
locked = json.loads(Path(f'data-provenance/{name}/source.lock.json').read_text())
root = Path('connectome_data') / name
root.mkdir(parents=True, exist_ok=True)

for filename, url in registry['files'].items():
    target = root / filename
    if not target.exists():
        print(f'[data] downloading {filename} from {url}', flush=True)
        partial = target.with_suffix(target.suffix + '.download')
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
    print(f'[data] verifying {filename}', flush=True)
    with target.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    expected = locked[filename]['sha256']
    if digest != expected:
        raise RuntimeError(f'Source checksum mismatch: {filename}: {digest} != {expected}')

(root / 'source.lock.json').write_text(json.dumps(locked, indent=2) + '\n')
print('[data] MaleCNS v1.0 source files verified', flush=True)
PY
}

prepare() {
  activate_env
  require_cmd clang++
  log "Downloading/verifying MaleCNS v1.0"
  download_and_verify_data
  log "Importing full retained connectome"
  python -m doom.connectome malecns_v1
  log "Preparing graph"
  python -m doom.prepare
  log "Auditing graph/data provenance"
  python -m doom.audit_data
  log "Building reviewed native neural kernel"
  python -m doom.build_kernel
  log "Building v6 plasticity kernel"
  python - <<'PY'
from doom_learning_v6.brain import build
print(build())
PY
  log "MaleCNS runtime ready"
}

smoke() {
  activate_env
  log "Running lightweight research decision/protocol tests"
  python -m pytest \
    tests/test_doom_learning_v6_conditioning.py \
    tests/test_doom_learning_v6_conditioning_strict.py \
    tests/test_doom_learning_v6_cue_screen.py \
    tests/test_doom_learning_v6_research_report.py \
    -q
}

assert_runtime_ready() {
  [[ -f outputs/doom/malecns_v1/graph.npz ]] || die "Prepared graph missing. Run prepare first."
  [[ -f outputs/doom/libneural.so || -f outputs/doom/libneural.dylib ]] || die "Native kernel missing. Run prepare first."
}

record_host() {
  local run_root="$1"
  mkdir -p "$run_root"
  {
    printf 'timestamp_utc='; date -u +'%Y-%m-%dT%H:%M:%SZ'
    printf 'git_commit='; git rev-parse HEAD 2>/dev/null || true
    printf 'git_branch='; git rev-parse --abbrev-ref HEAD 2>/dev/null || true
    printf 'python='; python --version 2>&1
    printf 'clang='; clang++ --version | head -n 1
    printf 'uname='; uname -a
    if command -v lscpu >/dev/null 2>&1; then lscpu; fi
    if command -v free >/dev/null 2>&1; then free -h; fi
  } > "$run_root/system-info.txt"
}

write_report() {
  local run_root="$1"
  python -m doom_learning_v6.research_report "$run_root"
}

preflight_passed() {
  local result="$1"
  python - "$result" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(0 if payload.get('conditioning_preflight_passed') else 1)
PY
}

baseline() {
  local run_id="${1:-$(date -u +'%Y%m%dT%H%M%SZ')}"
  activate_env
  assert_runtime_ready
  local run_root="$RUNS/$run_id"
  [[ ! -e "$run_root" ]] || die "Run directory already exists: $run_root (use a new run id)"
  mkdir -p "$run_root"
  record_host "$run_root"

  log "[$run_id] Stage 1/2: cue preflight"
  python -m doom_learning_v6.cue_screen --out "$run_root/cue-screen"
  write_report "$run_root"

  if ! preflight_passed "$run_root/cue-screen/results.json"; then
    log "[$run_id] No conditioning-ready cue pair. Strict training intentionally skipped."
    log "Report: $run_root/research-report.md"
    return 0
  fi

  log "[$run_id] Stage 2/2: strict differential conditioning"
  python -m doom_learning_v6.conditioning_strict \
    --preflight "$run_root/cue-screen/results.json" \
    --out "$run_root/conditioning-strict"
  write_report "$run_root"
  log "[$run_id] Baseline complete. Report: $run_root/research-report.md"
}

legacy_replication() {
  local run_id="$1"
  activate_env
  assert_runtime_ready
  local run_root="$RUNS/$run_id"
  [[ -d "$run_root" ]] || die "Run does not exist: $run_root"
  [[ ! -e "$run_root/conditioning-replication" ]] || die "Legacy replication output already exists"
  log "[$run_id] Running legacy v5 timing on v6 for controlled historical comparison"
  python -m doom_learning_v6.conditioning \
    --out "$run_root/conditioning-replication"
  write_report "$run_root"
}

full() {
  local run_id="${1:-$(date -u +'%Y%m%dT%H%M%SZ')}"
  baseline "$run_id"
  # Historical replication is useful even if the strict preflight blocks; it is
  # descriptive evidence about the v5->v6 model revision, not a substitute gate.
  legacy_replication "$run_id"
  log "[$run_id] Full research run complete"
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/snn_research_local.sh bootstrap
  bash scripts/snn_research_local.sh prepare
  bash scripts/snn_research_local.sh smoke
  bash scripts/snn_research_local.sh baseline [run-id]
  bash scripts/snn_research_local.sh legacy <run-id>
  bash scripts/snn_research_local.sh full [run-id]
  bash scripts/snn_research_local.sh report <run-id>
  bash scripts/snn_research_local.sh all [run-id]

Recommended first run in WSL2/Ubuntu:
  bash scripts/snn_research_local.sh all first-local

`all` performs bootstrap -> prepare -> smoke -> baseline -> historical replication.
Large MaleCNS source files are downloaded only when absent and are checksum-verified.
Experiment output is never overwritten; use a new run-id for each independent run.
EOF
}

command="${1:-}"
case "$command" in
  bootstrap) bootstrap ;;
  prepare) prepare ;;
  smoke) smoke ;;
  baseline) baseline "${2:-}" ;;
  legacy) [[ -n "${2:-}" ]] || die "legacy requires a run-id"; legacy_replication "$2" ;;
  full) full "${2:-}" ;;
  report)
    [[ -n "${2:-}" ]] || die "report requires a run-id"
    activate_env
    write_report "$RUNS/$2"
    ;;
  all)
    run_id="${2:-$(date -u +'%Y%m%dT%H%M%SZ')}"
    bootstrap
    prepare
    smoke
    full "$run_id"
    ;;
  -h|--help|help|'') usage ;;
  *) die "Unknown command: $command" ;;
esac
