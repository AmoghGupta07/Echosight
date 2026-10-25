#!/usr/bin/env bash
# Install the smart glasses as a service that starts when the Pi boots.
#   bash install_service.sh              install + enable (starts at next boot, or run: systemctl --user start smartglasses)
#   bash install_service.sh --uninstall  remove it
set -e
cd "$(dirname "$0")"
DIR="$(pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
ENV_FILE="$HOME/.config/smartglasses.env"

if [ "$1" = "--uninstall" ]; then
  systemctl --user disable --now smartglasses.service 2>/dev/null || true
  rm -f "$UNIT_DIR/smartglasses.service"
  systemctl --user daemon-reload
  echo "Removed. (Lingering is still on; undo with: sudo loginctl disable-linger $USER)"
  exit 0
fi

[ -x "$DIR/.venv/bin/python" ] || { echo "No .venv found. Run: bash setup.sh"; exit 1; }
[ -f "$DIR/voices/en_US-lessac-medium.onnx" ] || echo "Warning: voice file missing in $DIR/voices (setup.sh downloads it)"

# 1) permissions: earphone button (/dev/input), sensor (/dev/gpiochip*), audio
sudo usermod -aG input,gpio,audio "$USER"

# 2) the unit, with this folder's path filled in
mkdir -p "$UNIT_DIR"
sed "s|@INSTALL_DIR@|$DIR|g" smartglasses.service > "$UNIT_DIR/smartglasses.service"

# 3) secrets/settings file (created once, never overwritten)
if [ ! -f "$ENV_FILE" ]; then
  mkdir -p "$(dirname "$ENV_FILE")"
  cat > "$ENV_FILE" <<'ENV'
# Settings for the smart glasses service. Format: NAME=value  (no "export", no spaces around =)
GEMINI_API_KEY=
# GEMINI_MODEL=gemini-3.5-flash,gemini-2.5-flash
# SG_WARN_CM=30                 obstacle warning distance in cm
# SG_BUTTON_DEVICE=             only use input devices whose name contains this (e.g. your earphones' name)
# SG_SENSOR=0                   uncomment to disable the ultrasonic sensor
ENV
fi
chmod 600 "$ENV_FILE"

# 4) start at boot without a login
sudo loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable smartglasses.service

cat <<MSG

Installed.
  1) Put your key in:        nano $ENV_FILE        (GEMINI_API_KEY=...)
  2) REBOOT once:            sudo reboot           (new group membership: input, gpio, audio)
     After that it starts by itself on every boot.

Useful:
  systemctl --user status smartglasses        is it running?
  journalctl --user -u smartglasses -f        live log
  systemctl --user restart smartglasses       after editing code or $ENV_FILE
  systemctl --user stop smartglasses          stop (e.g. to run  python assistant.py  by hand)
MSG
