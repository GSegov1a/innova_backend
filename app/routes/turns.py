from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.conversations import create_conversation_turn


router = APIRouter()


@router.post("/turns")
def create_turn(
    session_id: int,
    role: str,
    text: str,
    db: Session = Depends(get_db),
):
    """Crea un turno de conversación para una sesión existente."""
    return create_conversation_turn(db, session_id, role, text)
