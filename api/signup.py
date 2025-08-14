from fastapi import FastAPI, Request, Response, Depends, HTTPException, APIRouter
from pydantic import BaseModel
import psycopg
from api.basemodel import Signup
from api.db import db_connection
import bcrypt

load = APIRouter()

@load.post("/signup")
async def signup(user: Signup):
    passwd = user.password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(passwd.encode('utf-8'), salt)
    passwd = hashed.decode('utf-8')
    conn = db_connection()
    print(bool(conn))
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE name = %s", (user.username,))
        existing = cursor.fetchone()
        if existing:
            raise HTTPException(status_code = 401, detail = "Username already exists")
        
        cursor.execute("INSERT INTO users (name, role, email_id, mobile, passwd) VALUES (%s,%s,%s,%s,%s)", (user.username, user.role, user.email, user.mobile, passwd))
        conn.commit()
        return {"Message" : "The Signup is Successful"}

    except psycopg.Error as error:
        raise HTTPException(status_code = 500, detail = f"MySQL Error : {error}")

    finally:
        cursor.close()
        conn.close()