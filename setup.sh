#!/usr/bin/env bash
# Setup for Raspberry Pi OS (64-bit), Debian or Ubuntu. Run from inside the project folder:
#   bash setup.sh              # USB webcam
#   bash setup.sh --picamera   # Pi Camera Module (also installs picamera2)
# On Windows/macOS see README.md section 1 instead.
set -e
cd "$(dirname "$0")"

sudo apt update
sudo apt install -y python3-venv python3-dev libportaudio2 portaudio19-dev \
                    libasound2-dev alsa-utils v4l-utils \
                    python3-gpiozero python3-lgpio python3-evdev

if [ "$1" = "--picamera" ]; then
  sudo apt install -y python3-picamera2
fi
# gpiozero/lgpio (ultrasonic sensor) and evdev (earphone button) come from apt, so the venv must
# be able to see system packages. Pip packages inside the venv still take precedence.
python3 -m venv --system-site-packages .venv

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Offline text-to-speech voice (about 60 MB)
mkdir -p voices
python -m piper.download_voices en_US-lessac-medium --data-dir voices

cat <<'MSG'

Setup finished. Next:
  1) Activate the environment each session:   source .venv/bin/activate
  2) Set your Gemini key (see README.md):      export GEMINI_API_KEY=$(cat ~/.gemini_key)
  3) Check devices:                            arecord -l ; aplay -l ; v4l2-ctl --list-devices
  4) Run the tests in order (README.md section 1) and finish with:  python assistant.py
  5) Start at boot:                                bash install_service.sh
MSG
