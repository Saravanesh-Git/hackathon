from pydantic import BaseModel

class Signup(BaseModel):
    username: str
    role: str
    email: str
    mobile: str
    password: str

class Login(BaseModel):
    username: str
    password: str