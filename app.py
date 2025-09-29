import os
import tempfile
import subprocess
import gradio as gr
from faster_whisper import WhisperModel

def extract_audio_to_16k(input_video_path: str, sample_rate: int = 16000) -> str:
    """Extracts mono 16kHz WAV via ffmpeg and returns the temp wav path."""
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def do_transcribe(video_file, model_size, device, compute_type, language, use_vad):
    if video_file is None:
        return "Please drop a video file.", None

    # Safe defaults for Apple Silicon: CPU + int8
    if device == "cpu" and compute_type == "int8_float16":
        compute_type = "int8"

    # Build model with a graceful compute-type fallback
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except ValueError as e:
        msg = str(e)
        # Fallbacks if requested type isn't supported by backend
        if "int8_float16" in msg:
            model = WhisperModel(model_size, device=device, compute_type="int8")
        else:
            model = WhisperModel(model_size, device=device, compute_type="float32")

    # Extract audio
    wav_path = extract_audio_to_16k(video_file)

    # Transcribe
    segments_gen, info = model.transcribe(
        wav_path,
        language=None if language.strip() == "" else language.strip(),
        vad_filter=use_vad,
        beam_size=1,
        temperature=0.0,
    )

    # Build plain-text transcript
    lines = []
    for seg in segments_gen:
        lines.append(seg.text.strip())
    transcript_text = "\n".join([l for l in lines if l])

    # Write to a temporary .txt for download
    out_fd, out_path = tempfile.mkstemp(prefix="transcript_", suffix=".txt")
    os.close(out_fd)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    # Cleanup audio
    try:
        os.remove(wav_path)
    except Exception:
        pass

    return transcript_text, out_path

with gr.Blocks(title="Quick Transcriber") as demo:
    gr.Markdown("## 🎙️ Drag & Drop Transcriber\nDrop a video, hit **Transcribe**, and view/download the transcript.")

    with gr.Row():
        video = gr.File(label="Drop video file here (mp4, mov, etc.)", file_types=[".mp4", ".mov", ".mkv", ".mp3", ".wav"])
    with gr.Accordion("Advanced settings", open=False):
        model = gr.Dropdown(choices=["tiny", "base", "small", "medium"], value="small", label="Model size")
        device = gr.Dropdown(choices=["cpu"], value="cpu", label="Device")  # keep CPU for M1 by default
        compute = gr.Dropdown(choices=["int8", "float32", "int8_float16"], value="int8", label="Compute type")
        language = gr.Textbox(value="", label="Force language (e.g., 'en' for English). Leave empty for auto-detect.")
        vad = gr.Checkbox(value=True, label="Voice Activity Detection (skip silences)")

    go = gr.Button("Transcribe", variant="primary")
    transcript = gr.Textbox(label="Transcript", lines=16, show_copy_button=True)
    download = gr.File(label="Download .txt")

    go.click(fn=do_transcribe, inputs=[video, model, device, compute, language, vad], outputs=[transcript, download])

if __name__ == "__main__":
    demo.launch()
