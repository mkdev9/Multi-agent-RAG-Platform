"""Tests for authentication endpoints."""
import pytest
from fastapi.testclient import TestClient


def test_register_user(client: TestClient):
    """Test user registration."""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "newuser@example.com",
            "password": "newpassword123",
            "full_name": "New User",
            "role": "user"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["email"] == "newuser@example.com"


def test_register_duplicate_email(client: TestClient, test_user):
    """Test registration with duplicate email."""
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": test_user.email,
            "password": "password123",
            "full_name": "Duplicate User"
        }
    )
    
    assert response.status_code == 409
    assert "already exists" in response.json()["error"]


def test_login_success(client: TestClient, test_user):
    """Test successful login."""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": test_user.email,
            "password": "testpassword123"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["user"]["email"] == test_user.email


def test_login_wrong_password(client: TestClient, test_user):
    """Test login with wrong password."""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": test_user.email,
            "password": "wrongpassword"
        }
    )
    
    assert response.status_code == 401


def test_login_nonexistent_user(client: TestClient):
    """Test login with nonexistent user."""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "nonexistent@example.com",
            "password": "password123"
        }
    )
    
    assert response.status_code == 401


def test_get_current_user(client: TestClient, auth_headers):
    """Test getting current user info."""
    response = client.get(
        "/api/v1/auth/me",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "email" in data
    assert "role" in data


def test_get_current_user_unauthorized(client: TestClient):
    """Test getting current user without auth."""
    response = client.get("/api/v1/auth/me")
    
    assert response.status_code == 401


def test_change_password(client: TestClient, auth_headers):
    """Test changing password."""
    response = client.post(
        "/api/v1/auth/change-password",
        headers=auth_headers,
        json={
            "current_password": "testpassword123",
            "new_password": "newtestpassword123"
        }
    )
    
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_change_password_wrong_current(client: TestClient, auth_headers):
    """Test changing password with wrong current password."""
    response = client.post(
        "/api/v1/auth/change-password",
        headers=auth_headers,
        json={
            "current_password": "wrongpassword",
            "new_password": "newtestpassword123"
        }
    )
    
    assert response.status_code == 401