from sqlalchemy.orm import Session

from app.models import ConversationSession, ConversationTurn


def get_active_session(db: Session, child_id: int):
    """Busca la sesión activa de un niño."""
    return (
        db.query(ConversationSession)
        .filter(
            ConversationSession.child_id == child_id,
            ConversationSession.status == "active",
        )
        .first()
    )


def get_or_create_active_session(db: Session, child_id: int):
    """Devuelve la sesión activa de un niño o crea una si no existe."""
    active_session = get_active_session(db, child_id)

    if active_session:
        return active_session

    session = ConversationSession(child_id=child_id)

    db.add(session)
    db.commit()
    db.refresh(session)

    return session


def create_conversation_turn(db: Session, session_id: int, role: str, text: str):
    """Guarda un turno de conversación y retorna el registro creado."""
    turn = ConversationTurn(
        session_id=session_id,
        role=role,
        text=text,
    )

    db.add(turn)
    db.commit()
    db.refresh(turn)

    return turn


def get_last_turns(db: Session, session_id: int, limit: int):
    """Obtiene los últimos turnos de una sesión en orden cronológico."""
    turns = (
        db.query(ConversationTurn)
        .filter(ConversationTurn.session_id == session_id)
        .order_by(ConversationTurn.created_at.desc())
        .limit(limit)
        .all()
    )

    return list(reversed(turns))
