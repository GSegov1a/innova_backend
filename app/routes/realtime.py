from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ConversationSession
from app.services.conversations import create_conversation_turn
from app.services.realtime import (
    build_realtime_session_config,
    create_openai_realtime_call,
    prepare_realtime_session,
    verify_device_token,
)


router = APIRouter(prefix="/realtime")


class RealtimeTurnCreate(BaseModel):
    """Payload para guardar un turno generado durante una sesión Realtime."""

    role: Literal["child", "assistant"]
    text: str


@router.post("/sessions/{child_id}/sdp")
async def create_realtime_sdp(
    child_id: int,
    request: Request,
    db: Session = Depends(get_db),
    x_device_token: str | None = Header(default=None),
):
    """Crea una sesión WebRTC Realtime y devuelve el SDP answer de OpenAI."""
    verify_device_token(x_device_token)

    sdp_offer = (await request.body()).decode("utf-8").strip()

    if not sdp_offer:
        raise HTTPException(status_code=400, detail="SDP offer is required")

    if not sdp_offer.startswith("v=0"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid SDP offer received. Length: {len(sdp_offer)}",
        )

    child, session, previous_turns = prepare_realtime_session(db, child_id)
    session_config = build_realtime_session_config(child, previous_turns)
    sdp_answer = await create_openai_realtime_call(sdp_offer, session_config)

    return Response(
        content=sdp_answer,
        media_type="application/sdp",
        headers={"X-Session-Id": str(session.id)},
    )


@router.post("/sessions/{session_id}/turns")
def create_realtime_turn(
    session_id: int,
    payload: RealtimeTurnCreate,
    db: Session = Depends(get_db),
    x_device_token: str | None = Header(default=None),
):
    """Guarda un turno reenviado por el Raspberry desde eventos Realtime."""
    verify_device_token(x_device_token)

    session = db.get(ConversationSession, session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    text = payload.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    return create_conversation_turn(db, session_id, payload.role, text)
