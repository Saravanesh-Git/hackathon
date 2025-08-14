from fastapi import FastAPI, Request, Response, Depends, HTTPException, APIRouter
from pydantic import BaseModel
import psycopg
from api.basemodel import Login
from api.db import db_connection
import bcrypt

load = APIRouter()

@load.post("/login")
async def signup(user: Login):
    passwd = user.password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(passwd.encode('utf-8'), salt)
    passwd = hashed.decode('utf-8')
    conn = db_connection()
    print(bool(conn))
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name, passwd FROM users WHERE name = %s", (user.username,))
        conn.commit()
        return {"Message" : "The Login is Successful"}

    except psycopg.Error as error:
        raise HTTPException(status_code = 500, detail = f"MySQL Error : {error}")

    finally:
        cursor.close()
        conn.close()