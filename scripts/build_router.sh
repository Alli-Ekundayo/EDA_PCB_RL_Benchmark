#!/usr/bin/env bash
# build_router.sh – Build the Rust pcb_router binary and install it to bin/
# Usage: bash scripts/build_router.sh [--force]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor/PcbRouter/pcb_router_rs"
BIN_DIR="$REPO_ROOT/bin"
BINARY="$VENDOR_DIR/target/release/pcb_router"
DEST="$BIN_DIR/pcb_router"

# ── Locate cargo ────────────────────────────────────────────────────────────
CARGO="${HOME}/.cargo/bin/cargo"
if ! command -v cargo &>/dev/null; then
  if [[ -x "$CARGO" ]]; then
    export PATH="${HOME}/.cargo/bin:$PATH"
  else
    echo "ERROR: cargo not found. Install Rust via https://rustup.rs" >&2
    exit 1
  fi
fi

echo "==> cargo version: $(cargo --version)"

# ── Check source exists ──────────────────────────────────────────────────────
if [[ ! -d "$VENDOR_DIR" ]]; then
  echo "==> Cloning PcbRouter repository..."
  mkdir -p "$REPO_ROOT/vendor"
  git -c http.version=HTTP/1.1 clone --depth=1 \
    https://github.com/Alli-Ekundayo/PcbRouter \
    "$REPO_ROOT/vendor/PcbRouter"
fi

# ── Skip build if binary is fresh and --force not supplied ──────────────────
if [[ -f "$DEST" && "${1:-}" != "--force" ]]; then
  echo "==> Binary already at $DEST (pass --force to rebuild)"
  "$DEST" 2>&1 | head -2 || true
  exit 0
fi

# ── Build ────────────────────────────────────────────────────────────────────
echo "==> Building pcb_router (release)..."
cargo build --release --manifest-path "$VENDOR_DIR/Cargo.toml"

# ── Install ──────────────────────────────────────────────────────────────────
mkdir -p "$BIN_DIR"
cp "$BINARY" "$DEST"
chmod +x "$DEST"

echo "==> Installed: $DEST"
echo "==> Smoke-test:"
"$DEST" 2>&1 | head -3 || true
