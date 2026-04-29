from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
import tempfile
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Child, ConversationSession, ConversationTurn

from datetime import datetime, timezone

import os
from dotenv import load_dotenv
from openai import OpenAI

from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import time

# crea las tablas automáticamente
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def root():
    return {"status": "ok"}





# Child endpoints
@app.post("/children")
def create_child(name: str, age: int, toy_name: str, db: Session = Depends(get_db)):
    child = Child(name=name, age=age, toy_name=toy_name)
    db.add(child)
    db.commit()
    db.refresh(child)
    return child

@app.get("/children")
def get_children(db: Session = Depends(get_db)):
    return db.query(Child).all()

@app.get("/children/{child_id}")
def get_child(child_id: int, db: Session = Depends(get_db)):
    return db.query(Child).get(child_id)





# ConversationSession endpoints
@app.post("/sessions")
def create_session(child_id: int, db: Session = Depends(get_db)):
    active_session = (
        db.query(ConversationSession)
        .filter(
            ConversationSession.child_id == child_id,
            ConversationSession.status == "active"
        )
        .first()
    )

    if active_session:
        return active_session

    session = ConversationSession(child_id=child_id)

    db.add(session)
    db.commit()
    db.refresh(session)

    return session


@app.get("/sessions/active/{child_id}")
def get_active_session(child_id: int, db: Session = Depends(get_db)):
    session = (
        db.query(ConversationSession)
        .filter(
            ConversationSession.child_id == child_id,
            ConversationSession.status == "active"
        )
        .first()
    )

    return session


@app.get("/sessions/{session_id}/turns")
def get_turns(session_id: int, db: Session = Depends(get_db)):
    session = db.query(ConversationSession).get(session_id)

    if not session:
        return {"error": "Session not found"}

    turns = (
        db.query(ConversationTurn)
        .filter(ConversationTurn.session_id == session_id)
        .order_by(ConversationTurn.created_at.asc())
        .all()
    )

    return turns



@app.post("/sessions/{session_id}/close")
def close_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(ConversationSession).get(session_id)

    if not session:
        return {"error": "Session not found"}

    session.status = "closed"
    session.ended_at = datetime.now(timezone.utc)

    db.commit()

    return {"status": "closed"}





# ConversationTurn endpoints
@app.post("/turns")
def create_turn(
    session_id: int,
    role: str,
    text: str,
    db: Session = Depends(get_db)
):
    turn = ConversationTurn(
        session_id=session_id,
        role=role,
        text=text
    )

    db.add(turn)
    db.commit()
    db.refresh(turn)

    return turn





# Endpoints Voz
def should_flush_tts(text: str) -> bool:
    text = text.strip()

    if not text:
        return False

    return text.endswith((".", "?", "!", "\n"))

@app.post("/voice-chat/{child_id}")
async def voice_chat(child_id: int, audio: UploadFile = File(...), db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    metrics = {}

    child = db.query(Child).get(child_id)

    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    session = (
        db.query(ConversationSession)
        .filter(
            ConversationSession.child_id == child_id,
            ConversationSession.status == "active"
        )
        .first()
    )

    if not session:
        session = ConversationSession(child_id=child_id)
        db.add(session)
        db.commit()
        db.refresh(session)

    suffix = audio.filename.split(".")[-1] if audio.filename else "wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as temp_audio:
        content = await audio.read()

        if not content:
            raise HTTPException(status_code=400, detail="Empty audio file")

        temp_audio.write(content)
        temp_audio_path = temp_audio.name

    t_stt_start = time.perf_counter()
    
    with open(temp_audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file
        )
    
    metrics["stt_seconds"] = round(time.perf_counter() - t_stt_start, 3)

    child_text = transcription.text

    child_turn = ConversationTurn(
        session_id=session.id,
        role="child",
        text=child_text
    )

    db.add(child_turn)
    db.commit()

    previous_turns = (
        db.query(ConversationTurn)
        .filter(ConversationTurn.session_id == session.id)
        .order_by(ConversationTurn.created_at.desc())
        .limit(8)
        .all()
    )

    previous_turns = list(reversed(previous_turns))

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
            )
        }
    ]

    for turn in previous_turns:
        if turn.role == "child":
            messages.append({"role": "user", "content": turn.text})
        elif turn.role == "assistant":
            messages.append({"role": "assistant", "content": turn.text})

    metrics["context_ready_seconds"] = round(time.perf_counter() - t0, 3)

    def audio_stream():
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

                if not first_tts_started:
                    metrics["first_sentence_ready_seconds"] = round(time.perf_counter() - t0, 3)
                    first_tts_started = True

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

        if sentence_buffer.strip():
            text_to_speak = sentence_buffer.strip()

            if not first_tts_started:
                metrics["first_sentence_ready_seconds"] = round(time.perf_counter() - t0, 3)
                first_tts_started = True

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

        assistant_turn = ConversationTurn(
            session_id=session.id,
            role="assistant",
            text=full_assistant_text
        )

        db.add(assistant_turn)
        db.commit()
        db.refresh(assistant_turn)

        metrics["total_backend_seconds"] = round(time.perf_counter() - t0, 3)

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

    return StreamingResponse(
    audio_stream(),
    media_type="audio/mpeg",
    headers={
        "X-Session-Id": str(session.id),
        }
    )