#!/usr/bin/env bash
# download_models.sh — run on a machine with internet access, then scp models/ to the air-gapped server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTS_DIR="$SCRIPT_DIR/models/tts"
STT_DIR="$SCRIPT_DIR/models/stt"

mkdir -p "$TTS_DIR" "$STT_DIR"

echo "=== Downloading TTS models (Piper) ==="

HF_PIPER="https://huggingface.co/rhasspy/piper-voices/resolve/main"

download_piper_voice() {
    local lang_path="$1"
    local filename="$2"
    echo "  -> $filename"
    curl -L --progress-bar \
        "$HF_PIPER/$lang_path/$filename" \
        -o "$TTS_DIR/$filename"
    curl -L --progress-bar \
        "$HF_PIPER/$lang_path/$filename.json" \
        -o "$TTS_DIR/$filename.json"
}

download_piper_voice "pl/pl_PL/darkman/medium" "pl_PL-darkman-medium.onnx"
download_piper_voice "pl/pl_PL/gosia/medium"   "pl_PL-gosia-medium.onnx"

echo ""
echo "=== Downloading STT model (faster-whisper large-v3) ==="
echo ""

# Method 1: huggingface-cli (preferred)
if command -v huggingface-cli &>/dev/null; then
    echo "Using huggingface-cli..."
    PYTHONIOENCODING=utf-8 huggingface-cli download Systran/faster-whisper-large-v3 \
        --local-dir "$STT_DIR"
elif command -v hf &>/dev/null; then
    echo "Using hf..."
    PYTHONIOENCODING=utf-8 hf download Systran/faster-whisper-large-v3 \
        --local-dir "$STT_DIR"
else
    echo ""
    echo "huggingface-cli not found. Manual steps:"
    echo "  1. pip install huggingface_hub"
    echo "  2. PYTHONIOENCODING=utf-8 huggingface-cli download Systran/faster-whisper-large-v3 \\"
    echo "       --local-dir $STT_DIR"
    echo "  OR visit: https://huggingface.co/Systran/faster-whisper-large-v3"
    echo "  and manually download all files into: $STT_DIR"
    exit 1
fi

echo ""
echo "=== Done! ==="
echo ""
echo "Model sizes:"
du -sh "$TTS_DIR" "$STT_DIR"
echo ""
echo "To copy to the air-gapped server:"
echo "  scp -r $SCRIPT_DIR/models/ user@server:/data/apps/voice-gateway/"
