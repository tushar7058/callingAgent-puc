#!/usr/bin/env sh
set -eu

KEY_FILE="${1:-key.json}"
KEY_BASENAME="$(basename "$KEY_FILE")"

if [ ! -f "$KEY_FILE" ]; then
  echo "Missing Google credentials file: $KEY_FILE" >&2
  exit 1
fi

lk agent update-secrets \
  --secret-mount "$KEY_FILE" \
  --secrets "GOOGLE_APPLICATION_CREDENTIALS=/etc/secrets/$KEY_BASENAME"
