import psycopg

def db_connection():
    return psycopg.connect(
    dbname = "test",
    user = "testing",
    password = "962006",
    host = "127.0.0.1"
 )