from fastapi import FastAPI, Request, Response, Depends, HTTPException, APIRouter
from pydantic import BaseModel
import psycopg
from api.signup import load as load_signup
from api.login import load as load_login
from api.detect import load as load_detect

app = FastAPI()

app.include_router(load_signup)
app.include_router(load_login)
app.include_router(load_detect)
