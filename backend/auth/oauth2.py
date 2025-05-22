from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
import os

# To be replaced with actual database models
# from db import get_db, models
# from models import User

from config import settings
from api.schemas import TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Function to create access token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    
    return encoded_jwt

# Function to verify token
async def verify_token(token: str):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        email: str = payload.get("sub")
        role: str = payload.get("role")
        
        if email is None:
            raise credentials_exception
            
        token_data = TokenData(email=email, role=role)
        return token_data
        
    except JWTError:
        raise credentials_exception

# Dependency to get current user from token
async def get_current_user(token: str = Depends(oauth2_scheme)):
    token_data = await verify_token(token)
    
    # This would be replaced with actual database query
    # user = get_user_by_email(token_data.email)
    # if user is None:
    #     raise credentials_exception
    
    # For now, we'll return the token data as a placeholder
    return token_data

# Dependency to verify user is active
async def get_current_active_user(current_user = Depends(get_current_user)):
    # Once we have user model
    # if not current_user.is_active:
    #     raise HTTPException(status_code=400, detail="Inactive user")
    
    return current_user

# Function to verify user role
def verify_role(required_roles: list):
    async def verify(current_user = Depends(get_current_active_user)):
        if current_user.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this resource"
            )
        return current_user
    return verify
