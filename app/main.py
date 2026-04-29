from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, engine
from app.routes import children, realtime, root, sessions, turns, voice


Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(root.router)
app.include_router(children.router)
app.include_router(sessions.router)
app.include_router(turns.router)
app.include_router(voice.router)
app.include_router(realtime.router)
