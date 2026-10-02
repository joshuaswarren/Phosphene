#!/usr/bin/env bash
# VOCAL SEPARATION for a2v "Listen to the voice only" — installed into the
# engine venv by install.js, by every Update (scripts/post_update.sh) and by
# the Lip-sync form's own Install button (POST /a2v/separator/install).
#
# WHY EVERY USER GETS IT. A2V drives a mouth from a waveform. Given the whole
# record, the audio encoder reads drums, bass and guitar as syllables and the
# model hedges: the mouth moves a little on everything and sings nothing.
# Conditioning on the separated vocal fixes that, and the panel muxes the
# original song back, so nothing the audience hears changes. The Lip-sync form
# has that box ticked by default — and until 4.17 nothing installed the
# separator, so every user who asked for the voice silently got the mix.
#
# WHAT IT COSTS. torch is already in this venv (the mflux image pack needs
# it), so this adds ~11 small packages (~15 MB) plus the htdemucs weights
# (80 MB, into mlx_models/demucs). It NEVER changes a package that is already
# installed: torch and numpy are pinned to the versions on disk, because a
# torchaudio resolve would otherwise drag torch to its own release and move the
# image engine with it.
#
# WHY NOT THE `demucs` CLI. It saves through torchaudio, which since 2.9 needs
# torchcodec — it dies after the whole separation. The panel runs
# scripts/a2v_separate.py instead (ffmpeg in, stdlib wave out).
#
# Idempotent: a second run on a ready install prints "ready" and exits 0.
# Exit non-zero = not installed; callers treat that as a WARN, never as a
# failed Install or Update.
#
# OUTCOME RECORD (4.17.4). Every exit writes last_install.json next to the
# weights: {"outcome", "error_class", "via", "ts"} - closed words only, no
# paths. The panel reads it at boot and reports it once (docs/ANALYTICS.md,
# separator_install), so the fleet can tell an Update that left no separator
# apart from one nobody ran. PHOSPHENE_SEPARATOR_VIA names the caller.
#
# Usage: a2v_stems_deps.sh <path to the ltx-2-mlx checkout>
set -uo pipefail

APP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SEP_HOME="${PHOSPHENE_SEPARATOR_HOME:-$APP_ROOT/mlx_models/demucs}"
VIA="${PHOSPHENE_SEPARATOR_VIA:-install}"
record() {
  mkdir -p "$SEP_HOME" 2>/dev/null || return 0
  printf '{"outcome":"%s","error_class":"%s","via":"%s","ts":%s}\n' \
    "$1" "$2" "$VIA" "$(date +%s)" > "$SEP_HOME/last_install.json" 2>/dev/null
  return 0
}

MLX_CHECKOUT="$(cd "${1:?ltx-2-mlx checkout required}" && pwd)" \
  || { record failed no_venv; exit 1; }

VENV=""
for cand in "$MLX_CHECKOUT/env" "$MLX_CHECKOUT/.venv"; do
  if [ -x "$cand/bin/python3.11" ] || [ -x "$cand/bin/python" ]; then
    VENV="$cand"
    break
  fi
done
if [ -z "$VENV" ]; then
  echo "no engine venv under $MLX_CHECKOUT - run the engine install first" >&2
  record failed no_venv
  exit 1
fi
PY="$VENV/bin/python3.11"
[ -x "$PY" ] || PY="$VENV/bin/python"

# The weights live with Phosphene's other weights, not in a torch cache. NOT
# ${TORCH_HOME:-...}: Pinokio's ENVIRONMENT may set TORCH_HOME for every app,
# and the panel looks in mlx_models/demucs (or where it tells us to).
export TORCH_HOME="$SEP_HOME"
unset PYTORCH_ENABLE_MPS_FALLBACK PYTORCH_MPS_FAST_MATH
RUNNER="$APP_ROOT/scripts/a2v_separate.py"
PROBE='import demucs.pretrained, demucs.apply, torch'

if "$PY" -c "$PROBE" >/dev/null 2>&1; then
  echo 'vocal separation: package already installed'
else
  echo 'Installing vocal separation (demucs, ~15 MB + 80 MB weights)...'
  PINS="$(mktemp -t phos_sep_pins)"
  "$PY" - > "$PINS" <<'PYPINS'
import importlib.metadata as m
for name in ("torch", "numpy"):
    try:
        print(f"{name}=={m.version(name)}")
    except m.PackageNotFoundError:
        pass
PYPINS
  echo "keeping: $(tr '\n' ' ' < "$PINS")"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" -c "$PINS" 'demucs==4.0.1'
  else
    "$PY" -m pip install -c "$PINS" 'demucs==4.0.1'
  fi
  rc=$?
  rm -f "$PINS"
  if [ $rc -ne 0 ]; then
    echo 'vocal separation: package install failed' >&2
    record failed pip_failed
    exit 1
  fi
  if ! "$PY" -c "$PROBE"; then
    echo 'vocal separation: installed but does not import' >&2
    record failed import_failed
    exit 1
  fi
fi

# Fetch the weights now, so the first lip-sync render does not stop to
# download them. Prove the runner loads the model, not just that pip ran.
if ! "$PY" "$RUNNER" --prefetch; then
  echo 'vocal separation: the model did not load (network?)' >&2
  echo 'It will download on first use instead.' >&2
  record failed weights_failed
  exit 1
fi
record ok ""
echo "vocal separation ready (weights in $TORCH_HOME)"
