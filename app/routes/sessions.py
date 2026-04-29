from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ConversationSession, ConversationTurn
from app.services.conversations import get_active_session as find_active_session
from app.services.conversations import get_or_create_active_session


router = APIRouter()


@router.post("/sessions")
def create_session(child_id: int, db: Session = Depends(get_db)):
    """Obtiene la sesión activa del niño o crea una nueva."""
    return get_or_create_active_session(db, child_id)


@router.get("/sessions/active/{child_id}")
def get_active_session(child_id: int, db: Session = Depends(get_db)):
    """Obtiene la sesión activa del niño, si existe."""
    return find_active_session(db, child_id)


@router.get("/sessions/{session_id}/turns")
def get_turns(session_id: int, db: Session = Depends(get_db)):
    """Lista los turnos de una sesión en orden cronológico."""
    session = db.get(ConversationSession, session_id)

    if not session:
        return {"error": "Session not found"}

    return (
        db.query(ConversationTurn)
        .filter(ConversationTurn.session_id == session_id)
        .order_by(ConversationTurn.created_at.asc())
        .all()
    )


@router.post("/sessions/{session_id}/close")
def close_session(session_id: int, db: Session = Depends(get_db)):
    """Marca una sesión como cerrada y registra su hora de término."""
    session = db.get(ConversationSession, session_id)

    if not session:
        return {"error": "Session not found"}

    session.status = "closed"
    session.ended_at = datetime.now(timezone.utc)

    db.commit()

    return {"status": "closed"}
