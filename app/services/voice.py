import os
import time

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def should_flush_tts(text: str) -> bool:
    """Indica si el buffer de texto ya puede enviarse a TTS."""
    text = text.strip()

    if not text:
        return False

    return text.endswith((".", "?", "!", "\n"))


def transcribe_audio(audio_path: str) -> str:
    """Transcribe un archivo de audio usando el modelo configurado."""
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
        )

    return transcription.text


def build_messages(child, previous_turns):
    """Construye los mensajes de contexto para la respuesta del asistente."""
    messages = [
        {
            "role": "system",
            "content": (
                f"Eres {child.toy_name}, un acompañante emocional para un niño llamado {child.name}. "
                "Responde como un amigo calmado. Usa frases cortas. "
                "No diagnostiques. No sermonees. No reemplaces al adulto. "
                "Si el niño está alterado, valida primero y guía con calma."
                "Responde máximo 4 frases"
                "Primera frase debe ser muy corta, 5 palabras maximo"
            ),
        }
    ]

    for turn in previous_turns:
        if turn.role == "child":
            messages.append({"role": "user", "content": turn.text})
        elif turn.role == "assistant":
            messages.append({"role": "assistant", "content": turn.text})

    return messages


def stream_assistant_audio(messages, metrics: dict, t0: float):
    """Genera audio por streaming desde la respuesta del LLM, frase por frase."""
    full_assistant_text = ""
    sentence_buffer = ""

    t_llm_start = time.perf_counter()
    first_llm_token_seen = False
    first_tts_started = False
    first_audio_chunk_sent = False

    llm_stream = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,
        stream=True,
    )

    for event in llm_stream:
        delta = event.choices[0].delta.content

        if not delta:
            continue

        if not first_llm_token_seen:
            metrics["llm_first_token_seconds"] = round(time.perf_counter() - t_llm_start, 3)
            first_llm_token_seen = True

        full_assistant_text += delta
        sentence_buffer += delta

        if should_flush_tts(sentence_buffer):
            text_to_speak = sentence_buffer.strip()
            sentence_buffer = ""

            for chunk in _stream_tts(
                text_to_speak,
                metrics,
                t0,
                first_tts_started,
                first_audio_chunk_sent,
            ):
                first_tts_started = True
                first_audio_chunk_sent = True
                yield chunk

    if sentence_buffer.strip():
        for chunk in _stream_tts(
            sentence_buffer.strip(),
            metrics,
            t0,
            first_tts_started,
            first_audio_chunk_sent,
        ):
            yield chunk

    return full_assistant_text


def _stream_tts(
    text_to_speak: str,
    metrics: dict,
    t0: float,
    first_tts_started: bool,
    first_audio_chunk_sent: bool,
):
    """Convierte un fragmento de texto a audio y produce sus chunks."""
    if not first_tts_started:
        metrics["first_sentence_ready_seconds"] = round(time.perf_counter() - t0, 3)

    t_tts_start = time.perf_counter()

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text_to_speak,
        response_format="mp3",
    ) as tts_response:
        for chunk in tts_response.iter_bytes(chunk_size=4096):
            if not first_audio_chunk_sent:
                metrics["tts_first_audio_chunk_seconds"] = round(time.perf_counter() - t_tts_start, 3)
                metrics["total_until_first_audio_chunk_seconds"] = round(time.perf_counter() - t0, 3)
                first_audio_chunk_sent = True

            yield chunk


def log_voice_metrics(metrics: dict, child_text: str, full_assistant_text: str):
    """Imprime en consola la línea de tiempo del procesamiento de voz."""
    print("\n" + "=" * 70)
    print("VOICE CHAT TIMELINE")
    print(f"\nChild said: {child_text}")
    print(f"Assistant: {full_assistant_text}\n")
    print("=" * 70)

    timeline = [
        ("STT done", metrics["stt_seconds"]),
        ("Context ready", metrics["context_ready_seconds"]),
        ("LLM first token", metrics["context_ready_seconds"] + metrics["llm_first_token_seconds"]),
        ("First sentence ready", metrics["first_sentence_ready_seconds"]),
        ("First audio sent", metrics["total_until_first_audio_chunk_seconds"]),
        ("Backend finished", metrics["total_backend_seconds"]),
    ]

    prev_time = 0.0

    for label, current_time in timeline:
        delta = current_time - prev_time
        print(f"[+{current_time:6.3f}s]  {label:<25}  (Δ {delta:5.3f}s)")
        prev_time = current_time

    print("=" * 70 + "\n")
