from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Child


router = APIRouter()


@router.post("/children")
def create_child(name: str, age: int, toy_name: str, db: Session = Depends(get_db)):
    """Crea un niño con su edad y nombre del juguete asociado."""
    child = Child(name=name, age=age, toy_name=toy_name)
    db.add(child)
    db.commit()
    db.refresh(child)
    return child


@router.get("/children")
def get_children(db: Session = Depends(get_db)):
    """Lista todos los niños registrados."""
    return db.query(Child).all()


@router.get("/children/{child_id}")
def get_child(child_id: int, db: Session = Depends(get_db)):
    """Obtiene un niño por su id."""
    return db.get(Child, child_id)
