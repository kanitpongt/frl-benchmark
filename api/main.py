# -*- coding: utf-8 -*-
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from routers import verify
import uvicorn

DESC_TEXT = "Face Verification Test"

app = FastAPI(
    title='Face Verification API',
    version="0.0.1",
    description=DESC_TEXT
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return {"Root Index"}


app.include_router(verify.router, prefix="/verify", tags=["Verify"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
