from fastapi import Request
from typing import Optional
from src.models.user import User

def get_current_user(request: Request) -> Optional[User]:
    token = request.headers.get("Authorization")
    if not token:
        return None

    #TODO: Token validation to be implemented
    return None
