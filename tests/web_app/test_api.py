import pytest
from fastapi.testclient import TestClient
from web_app.app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_create_connect4_game():
    response = client.post(
        "/games/new",
        json={"game_type": "connect4", "player_options": {"1": {"player_type": "human"}}}
    )
    assert response.status_code == 200
    data = response.json()
    assert "game_id" in data
    assert data["game_type"] == "connect4"
    return data["game_id"]

def test_get_connect4_state():
    game_id = test_create_connect4_game()
    response = client.get(f"/games/{game_id}/state")
    assert response.status_code == 200
    data = response.json()
    assert "state" in data
    assert "rows" in data
    assert "columns" in data
    assert "legal_actions" in data
    assert len(data["state"]) == 6 # Rows
    assert len(data["state"][0]) == 7 # Cols

def test_create_othello_game():
    response = client.post(
        "/games/new",
        json={"game_type": "othello", "player_options": {"1": {"player_type": "human"}}}
    )
    assert response.status_code == 200
    data = response.json()
    assert "game_id" in data
    assert data["game_type"] == "othello"
    return data["game_id"]

def test_get_othello_state():
    game_id = test_create_othello_game()
    response = client.get(f"/games/{game_id}/state")
    assert response.status_code == 200
    data = response.json()
    assert "state" in data
    assert "rows" in data
    assert "columns" in data
    assert "legal_actions" in data
    assert data["rows"] == 8
    assert data["columns"] == 8

def test_make_move_othello():
    game_id = test_create_othello_game()
    # Fetch state to get a valid legal action (Othello uses specific coordinates)
    state_response = client.get(f"/games/{game_id}/state")
    state_data = state_response.json()
    legal_actions = state_data["legal_actions"]
    assert len(legal_actions) > 0
    
    # Pick first legal action
    # Legal actions in Othello are tuples [row, col]
    move_row, move_col = legal_actions[0]
    
    response = client.post(
        f"/games/{game_id}/move",
        json={"row": move_row, "col": move_col}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
