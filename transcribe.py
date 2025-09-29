#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import timedelta

from faster_whisper import WhisperModel

def hms(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    ms = int((td.total_seconds() - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

def write_txt(segments, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg["text"].strip() + "\n")
    return out_path

def extract_audio(input_path, tmpdir, sample_rate=16000):
    wav_path = os.path.join(tmpdir, "audio_16k.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        wav_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_path
    except FileNotFoundError:
        print("Error: ffmpeg not found. Install ffmpeg and ensure it is on PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("ffmpeg failed to extract audio.", file=sys.stderr)
        sys.exit(1)

def transcribe(
    audio_path,
    model_size="small",
    device="auto",
    compute_type="int8",
    language=None,
    vad_filter=True,
    beam_size=1,
    temperature=0.0,
):
    # make int8 the safe default on CPU
    if device.lower() == "cpu" and compute_type == "int8_float16":
        compute_type = "int8"

    # try to build model; if compute type not supported, fallback smartly
    try:
        model = WhisperModel(model_size_or_path=model_size, device=device, compute_type=compute_type)
    except ValueError as e:
        msg = str(e)
        if "int8_float16" in msg:
            compute_type = "int8"
        else:
            compute_type = "float32"
        model = WhisperModel(model_size_or_path=model_size, device=device, compute_type=compute_type)

    segments_generator, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=vad_filter,
        beam_size=beam_size,
        temperature=temperature,
        # word_timestamps=True,  # enable if you want per-word times
    )

    segs = []
    for seg in segments_generator:
        segs.append({
            "start": float(seg.start) if seg.start else 0.0,
            "end": float(seg.end) if seg.end else 0.0,
            "text": seg.text.strip(),
        })
    return segs

def main():
    parser = argparse.ArgumentParser(description="Fast local video transcription with faster-whisper.")
    parser.add_argument("input", help="Path to video/audio file")
    parser.add_argument("--out", help="Output base path (without extension). Default: transcribed_videos/<name>", default=None)
    parser.add_argument("--model", default="small",
                        help="Whisper model size/path (tiny, base, small, medium, large-v3, or local path). Default: small")
    parser.add_argument("--device", default="auto",
                        help='Device: "auto", "cuda", or "cpu". Default: auto')
    parser.add_argument("--compute-type", default="int8",
                        help='Compute type: "int8" (CPU), "int8_float16" (GPU), "float16", "float32". Default: int8')
    parser.add_argument("--language", default=None, help="Language code (e.g. en, fr). Default: auto-detect")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filter")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size; 1 is greedy and fastest.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Output dir: transcribed_videos next to this script
    out_dir = os.path.join(os.path.dirname(__file__), "transcribed_videos")
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    base_out = args.out or os.path.join(out_dir, base_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = extract_audio(args.input, tmpdir, sample_rate=16000)
        segments = transcribe(
            wav_path,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            vad_filter=(not args.no_vad),
            beam_size=args.beam_size,
            temperature=args.temperature,
        )

    out_path = write_txt(segments, base_out + ".txt")
    print("Done.")
    print("  ->", out_path)

if __name__ == "__main__":
    main()
