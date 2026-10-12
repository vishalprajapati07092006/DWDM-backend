"""
auth.py
=======
JWT authentication + password hashing for the donor DSS.

- Users are stored in the SAME data_warehouse.db (table: users)
- Passwords are hashed with bcrypt (via passlib)
- Tokens are signed with HS256 JWT
- Default admin seeded on first boot: admin / admin123
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from ml_engine import DB_PATH

# ------------ Config ------------
# ⚠️ In production, load this from an environment variable
SECRET_KEY = "change-me-to-a-long-random-secret-string-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ------------ Schemas ------------
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class UserOut(BaseModel):
    id: int
    username: str


# ------------ DB helpers ------------
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_users_table():
    """Create the users table and seed a default admin (only once)."""
    conn = _get_conn()
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("SELECT COUNT(*) AS n FROM users")
        if c.fetchone()["n"] == 0:
            c.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                ("admin", pwd_context.hash("admin123")),
            )
            print("👤 Seeded default user → admin / admin123")

        conn.commit()
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[dict]:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ------------ Password + JWT ------------
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


# ------------ Dependency: current user from Bearer token ------------
def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise cred_exc
    except JWTError:
        raise cred_exc

    user = get_user_by_username(username)
    if user is None:
        raise cred_exc
    return user