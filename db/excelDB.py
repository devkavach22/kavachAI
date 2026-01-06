from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv

load_dotenv()

# Database credentials and connection details
DB_USER = os.getenv("DB_USER")
# Encode password to safely include special characters
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")  # MySQL server host
DB_PORT = os.getenv("DB_PORT")  # Default MySQL port
DB_NAME = os.getenv("DB_NAME")

# Construct the SQLAlchemy database URI using the PyMySQL driver
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

sqlEngine = None


def connect_to_db():
    """Create an engine, ensure the target database exists, and return the engine.

    The function first connects to the MySQL server without specifying a database
    to run a `CREATE DATABASE IF NOT EXISTS` statement. It then creates the
    engine for the specific database and returns it.
    """
    # Server‑level URL (no database) – used to create the database if missing
    server_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/"
    server_engine = create_engine(server_url)

    # create database if not exists
    with server_engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
        print(f"Database '{DB_NAME}' ensured to exist.")
    # Engine for the specific database
    global sqlEngine
    sqlEngine = create_engine(DATABASE_URL)
    return sqlEngine


# Optional: run a quick connection test when the module is executed directly
if __name__ == "__main__":
    connect_to_db()
