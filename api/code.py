from fastapi import FastAPI, Request, Response, Depends, HTTPException, APIRouter
from pydantic import BaseModel
import psycopg
from api.db import db_connection

load = APIRouter()

@load.get("/code/get")
async def get_code():
    conn = db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT ssh, passwd FROM users WHERE user_id = %s", (user_id,))
        existing = cursor.fetchone()
        

    finally:
        cursor.close()
        conn.close()