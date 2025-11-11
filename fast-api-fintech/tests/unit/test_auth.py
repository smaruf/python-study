"""Unit tests for authentication endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_register_user():
    """Test user registration."""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "newuser@example.com",
            "password": "SecurePass123!",
            "full_name": "New User"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@example.com"
    assert data["full_name"] == "New User"
    assert "id" in data
    assert "password" not in data  # Password should not be in response


def test_register_user_invalid_email():
    """Test user registration with invalid email."""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "invalid-email",
            "password": "SecurePass123!",
            "full_name": "New User"
        }
    )
    assert response.status_code == 422  # Validation error


def test_login():
    """Test user login."""
    response = client.post(
        "/api/v1/auth/login",
        params={
            "email": "test@example.com",
            "password": "SecurePass123!"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


def test_refresh_token():
    """Test token refresh."""
    # First login to get refresh token
    login_response = client.post(
        "/api/v1/auth/login",
        params={
            "email": "test@example.com",
            "password": "SecurePass123!"
        }
    )
    refresh_token = login_response.json()["refresh_token"]
    
    # Use refresh token to get new access token
    response = client.post(
        "/api/v1/auth/refresh",
        params={"refresh_token": refresh_token}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_logout():
    """Test user logout."""
    response = client.post("/api/v1/auth/logout")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
