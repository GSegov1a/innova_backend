import json
import logging
import os

import httpx
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import Child
from app.services.conversations import get_last_turns, get_or_create_active_session


OPENAI_REALTIME_CALLS_URL = "https://api.openai.com/v1/realtime/calls"
LOGGER = logging.getLogger("innova-realtime")


def get_realtime_model() -> str:
    """Obtiene el modelo Realtime configurado para nuevas sesiones."""
    return os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-mini")


def get_realtime_voice() -> str:
    """Obtiene la voz configurada para las respuestas de Realtime."""
    return os.getenv("OPENAI_REALTIME_VOICE", "alloy")


def get_realtime_transcription_model() -> str:
    """Obtiene el modelo usado para transcribir el audio entrante."""
    return os.getenv("OPENAI_REALTIME_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")


def get_realtime_noise_reduction() -> str:
    """Obtiene el tipo de reducción de ruido para la entrada de audio."""
    return os.getenv("OPENAI_REALTIME_NOISE_REDUCTION", "near_field")


def get_vad_threshold() -> float:
    """Obtiene el umbral de activación VAD; más alto exige voz más fuerte."""
    return float(os.getenv("OPENAI_REALTIME_VAD_THRESHOLD", "0.75"))


def get_vad_silence_duration_ms() -> int:
    """Obtiene cuánto silencio se espera antes de cerrar el turno."""
    return int(os.getenv("OPENAI_REALTIME_VAD_SILENCE_MS", "900"))


def get_vad_prefix_padding_ms() -> int:
    """Obtiene cuánto audio previo al inicio de voz se conserva."""
    return int(os.getenv("OPENAI_REALTIME_VAD_PREFIX_MS", "300"))


def get_vad_interrupt_response() -> bool:
    """Obtiene si una nueva voz debe interrumpir respuestas en curso."""
    return os.getenv("OPENAI_REALTIME_VAD_INTERRUPT_RESPONSE", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def verify_device_token(device_token: str | None):
    """Valida que el Raspberry esté autorizado para usar endpoints Realtime."""
    expected_token = os.getenv("RASPBERRY_DEVICE_TOKEN")

    if not expected_token:
        raise HTTPException(status_code=500, detail="Raspberry device token is not configured")

    if not device_token or device_token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid device token")


def prepare_realtime_session(db: Session, child_id: int):
    """Valida el niño y obtiene la sesión activa para una conexión Realtime."""
    child = db.get(Child, child_id)

    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    session = get_or_create_active_session(db, child_id)
    previous_turns = get_last_turns(db, session.id, limit=8)

    return child, session, previous_turns


def build_realtime_session_config(child: Child, previous_turns) -> dict:
    """Construye la configuración de sesión que se envía a OpenAI Realtime."""
    return {
        "type": "realtime",
        "model": get_realtime_model(),
        "instructions": build_realtime_instructions(child, previous_turns),
        "audio": {
            "input": {
                "transcription": {
                    "model": get_realtime_transcription_model(),
                },
                "noise_reduction": {
                    "type": get_realtime_noise_reduction(),
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": get_vad_threshold(),
                    "prefix_padding_ms": get_vad_prefix_padding_ms(),
                    "silence_duration_ms": get_vad_silence_duration_ms(),
                    "create_response": True,
                    "interrupt_response": get_vad_interrupt_response(),
                },
            },
            "output": {
                "voice": get_realtime_voice(),
            },
        },
    }


def build_realtime_instructions(child: Child, previous_turns) -> str:
    """Construye el contexto breve que recibirá el modelo Realtime."""
    instructions = (
        f"Eres {child.toy_name}, un acompañante emocional para un niño llamado {child.name}, "
        f"de {child.age} años. Responde en español como un amigo calmado. "
        "Usa frases cortas. No diagnostiques. No sermonees. No reemplaces al adulto. "
        "Si el niño está alterado, valida primero y guía con calma. "
        "Responde máximo 4 frases. La primera frase debe ser muy corta, 5 palabras máximo."
    )

    if not previous_turns:
        return instructions

    history = "\n".join(
        f"{turn.role}: {turn.text}"
        for turn in previous_turns
    )

    return f"{instructions}\n\nContexto reciente de la conversación:\n{history}"


def log_realtime_session_config(session_config: dict):
    """Registra la configuración Realtime efectiva sin secretos."""
    audio = session_config.get("audio", {})
    audio_input = audio.get("input", {})
    turn_detection = audio_input.get("turn_detection", {})
    noise_reduction = audio_input.get("noise_reduction", {})
    transcription = audio_input.get("transcription", {})
    audio_output = audio.get("output", {})

    LOGGER.info(
        "Realtime config: model=%s voice=%s transcription=%s noise_reduction=%s "
        "vad_type=%s vad_threshold=%s vad_prefix_ms=%s vad_silence_ms=%s interrupt_response=%s",
        session_config.get("model"),
        audio_output.get("voice"),
        transcription.get("model"),
        noise_reduction.get("type"),
        turn_detection.get("type"),
        turn_detection.get("threshold"),
        turn_detection.get("prefix_padding_ms"),
        turn_detection.get("silence_duration_ms"),
        turn_detection.get("interrupt_response"),
    )


async def create_openai_realtime_call(sdp_offer: str, session_config: dict) -> str:
    """Intercambia el SDP offer local por un SDP answer de OpenAI Realtime."""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured")

    log_realtime_session_config(session_config)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            OPENAI_REALTIME_CALLS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={
                "sdp": (None, sdp_offer, "application/sdp"),
                "session": (None, json.dumps(session_config), "application/json"),
            },
        )

    if response.status_code >= 400:
        detail = {
            "openai_error": response.text[:500] or "OpenAI Realtime call failed",
            "sdp_offer_length": len(sdp_offer.encode("utf-8")),
            "sdp_offer_first_line": sdp_offer.splitlines()[0] if sdp_offer else "",
            "sdp_offer_ends_with_crlf": sdp_offer.endswith("\r\n"),
        }
        raise HTTPException(status_code=502, detail=detail)

    return response.text
