from fastapi import APIRouter


router = APIRouter()


@router.get("/")
def root():
    """Responde el estado básico del backend."""
    return {"status": "ok"}
