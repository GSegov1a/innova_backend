import os
import tempfile
import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.services.conversations import (
    create_conversation_turn,
    get_last_turns,
    get_or_create_active_session,
)
from app.database import get_db
from app.models import Child
from app.services.voice import (
    build_messages,
    log_voice_metrics,
    stream_assistant_audio,
    transcribe_audio,
)


router = APIRouter()


@router.post("/voice-chat/{child_id}")
async def voice_chat(child_id: int, audio: UploadFile = File(...), db: Session = Depends(get_db)):
    """Procesa un audio del niño y responde con audio generado por streaming."""
    t0 = time.perf_counter()
    metrics = {}

    child = db.get(Child, child_id)

    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    session = get_or_create_active_session(db, child_id)
    temp_audio_path = await _save_upload_to_temp_file(audio)

    try:
        t_stt_start = time.perf_counter()
        child_text = transcribe_audio(temp_audio_path)
        metrics["stt_seconds"] = round(time.perf_counter() - t_stt_start, 3)
    finally:
        os.unlink(temp_audio_path)

    create_conversation_turn(db, session.id, "child", child_text)

    previous_turns = get_last_turns(db, session.id, limit=8)
    messages = build_messages(child, previous_turns)
    metrics["context_ready_seconds"] = round(time.perf_counter() - t0, 3)

    def audio_stream():
        """Genera chunks de audio y guarda la respuesta final del asistente."""
        full_assistant_text = yield from stream_assistant_audio(messages, metrics, t0)

        create_conversation_turn(db, session.id, "assistant", full_assistant_text)
        metrics["total_backend_seconds"] = round(time.perf_counter() - t0, 3)
        log_voice_metrics(metrics, child_text, full_assistant_text)

    return StreamingResponse(
        audio_stream(),
        media_type="audio/mpeg",
        headers={"X-Session-Id": str(session.id)},
    )


async def _save_upload_to_temp_file(audio: UploadFile) -> str:
    """Guarda el upload de audio en un archivo temporal y retorna su ruta."""
    suffix = audio.filename.split(".")[-1] if audio.filename else "wav"
    content = await audio.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as temp_audio:
        temp_audio.write(content)
        return temp_audio.name
