# web_app/main.py

import logging
from datetime import datetime
from threading import Lock
from typing import Any, cast
import numpy as np

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Imports from rgi3-4 structure
from rgi.rgizero.games.base import Game
from rgi.rgizero.games.game_registry import create_game, list_games
from rgi.rgizero.players.base import Player
from rgi.rgizero.players.human_player import HumanPlayer
from rgi.rgizero.players.random_player import RandomPlayer
from rgi.rgizero.players.minimax_player import MinimaxPlayer
from rgi.rgizero.players.alphazero import AlphazeroPlayer
from rgi.rgizero.evaluators import UniformEvaluator
from rgi.rgizero.serving.server_manager import ModelServerManager
from rgi.rgizero.serving.grpc_evaluator import GrpcEvaluator
from rgi.rgizero.data.trajectory_dataset import Vocab
from rgi.rgizero.common import TOKENS
from pathlib import Path

print("Server restarted at", datetime.now())

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="web_app/templates")
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

# Configuration flags
VERBOSE_AI_LOGGING = True  # Set to False to disable AI score logging during games

# Initialize Server Manager
server_manager = ModelServerManager()

@app.on_event("shutdown")
async def shutdown_event():
    server_manager.shutdown_all()


class ThreadSafeCounter:
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    def current(self) -> int:
        with self._lock:
            return self._value


# In-memory storage for game sessions
GameSession = dict[str, Any]
games: dict[int, GameSession] = {}
game_counter = ThreadSafeCounter()

# Simple player registry for web app (can be expanded)
WEB_PLAYER_REGISTRY = {
    "human": HumanPlayer,
    "random": RandomPlayer,
    "minimax": MinimaxPlayer,
    "zerozero": AlphazeroPlayer,
}


def serialize_obj(obj: Any) -> Any:
    """Recursively serialize objects including dataclasses and numpy arrays."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [serialize_obj(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: serialize_obj(v) for k, v in obj.items()}
    elif hasattr(obj, "__dataclass_fields__"):
        return {k: serialize_obj(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return obj


def serialize_state(state: Any) -> dict[str, Any]:
    return serialize_obj(state)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})


@app.get("/api/models")
async def list_models() -> dict[str, list[str]]:
    """List available model files from experiments directory."""
    models: list[str] = []
    experiments_dir = Path("experiments")
    if experiments_dir.exists():
        for pt_file in experiments_dir.rglob("*.pt"):
            # Only include final generation models (gen-N.pt, not ckpt.pt or best.pt in subdirs)
            if pt_file.name.startswith("gen-") and pt_file.name.endswith(".pt"):
                # Avoid duplicates from subdirectories
                if pt_file.parent.name == "models":
                    models.append(str(pt_file))
    # Sort by path for consistent ordering
    models.sort()
    return {"models": models}


@app.post("/games/new")
async def create_new_game(request: Request) -> dict[str, Any]:
    data = await request.json()
    game_type: str = data.get("game_type", "")
    game_options: dict[str, Any] = data.get("game_options", {})
    player_options: dict[int, dict[str, Any]] = {int(k): v for k, v in data.get("player_options", {}).items()}
    logger.info(
        "Creating new game. Type: %s, Game Options: %s, Player Options: %s",
        game_type,
        game_options,
        player_options,
    )

    if game_type not in list_games():
        logger.error("Invalid game type: %s", game_type)
        raise HTTPException(status_code=400, detail=f"Invalid game type. Available: {list_games()}")

    # Create game instance
    game = create_game(game_type, **game_options)
    state = game.initial_state()
    game_id = game_counter.increment()

    # Initialize players
    players: dict[int, Player[Any, Any]] = {}
    for player_id, options in player_options.items():
        p_type = options.get("player_type", "human")
        constructor_options = {k: v for k, v in options.items() if k != "player_type"}
        
        # Handle AlphaZero variants specially
        if p_type in ("zerozero", "zerozero_best", "alphazero_custom"):
            sims = constructor_options.get("simulations", 50)  # Default 50 for fast play
            temp = constructor_options.get("temperature", 0.0)  # Default 0.0 (greedy)
            
            if p_type == "zerozero":
                # Untrained: use uniform evaluator (no model needed)
                num_players = game.num_players(state)
                evaluator = UniformEvaluator(num_players=num_players)
                logger.info(f"Setting up ZeroZero (Untrained) with uniform evaluator, {sims} simulations, temp={temp}")
                players[player_id] = AlphazeroPlayer(game=game, evaluator=evaluator, simulations=sims, temperature=temp)
                
            elif p_type in ("zerozero_best", "alphazero_custom"):
                # Get model path
                if p_type == "zerozero_best":
                    model_path = "experiments/overnight_2026_01_08/06_combined/06_combined/models/gen-4.pt"
                else:
                    model_path = constructor_options.get("model_path", "")
                    if not model_path:
                        raise HTTPException(status_code=400, detail="alphazero_custom requires model_path")
                
                logger.info(f"Setting up AlphaZero with model: {model_path}, {sims} simulations, temp={temp}")
                
                # Get/Start gRPC server
                try:
                    port = server_manager.get_port(model_path, game_type)
                    logger.info(f"Model server running on port {port}")
                    
                    # Create Vocab for encoder
                    base_game_ref = game.base_game if hasattr(game, "base_game") else game
                    vocab = Vocab(itos=[TOKENS.START_OF_GAME] + list(base_game_ref.all_actions()))

                    # Create evaluator connected to this port
                    evaluator = GrpcEvaluator(
                        host="localhost",
                        port=port,
                        vocab=vocab,
                        vocab_size=vocab.vocab_size
                    )
                    await evaluator.connect()
                    
                    players[player_id] = AlphazeroPlayer(
                        game=game, evaluator=evaluator, simulations=sims, temperature=temp
                    )
                except Exception as e:
                    logger.error(f"Failed to start model server: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to load AI model: {e}")
            continue
        
        # Handle other player types via registry
        if p_type not in WEB_PLAYER_REGISTRY:
            logger.warning(f"Unknown player type '{p_type}', defaulting to human")
            p_type = "human"
        
        player_cls = WEB_PLAYER_REGISTRY[p_type]
        try:
            import inspect
            sig = inspect.signature(player_cls.__init__)
            call_args = constructor_options.copy()

            if "game" in sig.parameters:
                call_args["game"] = game
            if "player_id" in sig.parameters:
                call_args["player_id"] = player_id

            players[player_id] = player_cls(**call_args)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to initialize player {player_id} ({p_type}): {e}")
            # Fallback for simple players
            try:
                players[player_id] = player_cls()
            except Exception:
                raise HTTPException(status_code=500, detail=f"Failed to initialize '{p_type}': {e}")

    logger.debug("Players initialized: %s", {pid: type(p).__name__ for pid, p in players.items()})

    # Store game session
    games[game_id] = {
        "game": game,
        "state": state,
        "players": players,
        "game_options": game_options,
        "player_options": player_options,
    }
    logger.info(
        "New game created. ID: %d, Game Options: %s, Player Options: %s",
        game_id,
        game_options,
        player_options,
    )
    return {"game_id": game_id, "game_type": game_type}


@app.get("/games/{game_id}/state")
async def get_game_state(game_id: int) -> dict[str, Any]:
    logger.debug("Fetching game state for game ID: %d", game_id)
    game_session = games.get(game_id)
    if not game_session:
        logger.error("Game not found. ID: %d", game_id)
        raise HTTPException(status_code=404, detail="Game not found")

    game = cast(Game[Any, Any], game_session["game"])
    state = game_session["state"]

    # Serialize state recursively first
    serialized_state = serialize_obj(state)

    # Flatten structure for frontend
    response_data = {}

    # Handle HistoryTrackingState wrapping
    base_state = (
        serialized_state.get("base_state", serialized_state) if isinstance(serialized_state, dict) else serialized_state
    )

    # Extract core fields
    # Connect4 frontend expects 'state' to be the board array
    if isinstance(base_state, dict) and "board" in base_state:
        response_data["state"] = base_state["board"]
        response_data["current_player"] = base_state.get("current_player")
        response_data["is_terminal"] = base_state.get("is_terminal")
        response_data["winner"] = base_state.get("winner")  # May be None
    else:
        # Fallback or generic state
        response_data["state"] = serialized_state
        response_data["current_player"] = game.current_player_id(state)
        response_data["is_terminal"] = game.is_terminal(state)
        response_data["winner"] = None
    
    # Calculate winner from reward() if game is terminal but winner not in state
    # (Othello doesn't have a winner field in state, it's computed from piece counts)
    if game.is_terminal(state) and response_data.get("winner") is None:
        winner = None
        for pid in game.player_ids(state):
            if game.reward(state, pid) == 1.0:
                winner = pid
                break
        response_data["winner"] = winner

    # Add Dimensions (Frontend expects 'rows' and 'columns')
    # Unwrap game to find base game with dimensions if needed
    current_game_obj = game
    while hasattr(current_game_obj, "base_game"):
        current_game_obj = current_game_obj.base_game

    if hasattr(current_game_obj, "height"):
        response_data["rows"] = current_game_obj.height
    if hasattr(current_game_obj, "width"):
        response_data["columns"] = current_game_obj.width
    # Fallback for Othello or square games
    if "rows" not in response_data and hasattr(current_game_obj, "board_size"):
        response_data["rows"] = current_game_obj.board_size
        response_data["columns"] = current_game_obj.board_size

    # Serialize legal actions for the frontend to highlight
    response_data["legal_actions"] = serialize_obj(game.legal_actions(state))

    # Add options
    response_data["game_options"] = game_session["game_options"]
    response_data["player_options"] = game_session["player_options"]

    # Keep full serialized state for debugging or other frontends
    response_data["full_state"] = serialized_state

    logger.debug("Game state for game %d: %s", game_id, response_data)
    return response_data


@app.post("/games/{game_id}/move")
async def make_move(game_id: int, action_data: dict[str, Any]) -> dict[str, Any]:
    game_session = games.get(game_id)
    if not game_session:
        raise HTTPException(status_code=404, detail="Game not found")

    game = cast(Game[Any, Any], game_session["game"])
    state = game_session["state"]

    try:
        # TODO: Better action parsing based on game type
        # For Connect4, action is just column integer
        if "column" in action_data:
            action = int(action_data["column"])
        elif "row" in action_data and "col" in action_data:
            action = (int(action_data["row"]), int(action_data["col"]))
        else:
            # Try to infer or use header
            # Fallback for generic action
            # Assuming single value action if not structured
            action = list(action_data.values())[0]

        if action not in game.legal_actions(state):
            return {"success": False, "error": f"Invalid move {action}. Legal: {game.legal_actions(state)}"}

        new_state = game.next_state(state, action)
        game_session["state"] = new_state
        return {"success": True}
    except ValueError as ve:
        return {"success": False, "error": str(ve)}
    except Exception as e:
        logger.error(f"Error making move: {e}")
        return {"success": False, "error": str(e)}


@app.post("/games/{game_id}/ai_move")
async def make_ai_move(game_id: int) -> dict[str, Any]:
    logger.debug("Attempting AI move for game ID: %d", game_id)
    game_session = games.get(game_id)
    if not game_session:
        logger.error("Game not found. ID: %d", game_id)
        raise HTTPException(status_code=404, detail="Game not found")

    game = cast(Game[Any, Any], game_session["game"])
    state = game_session["state"]
    players = cast(dict[int, Player[Any, Any]], game_session["players"])

    current_player_id = game.current_player_id(state)
    current_player = players.get(current_player_id)

    if not current_player:
        return {"success": False, "reason": "Player not found"}

    if game.is_terminal(state):
        logger.info("Game %d is already in a terminal state.", game_id)
        return {"success": False, "reason": "Game is already over"}

    if isinstance(current_player, HumanPlayer):
        logger.info("Current player %d is human. No AI move made.", current_player_id)
        return {"success": False, "reason": "Current player is human"}

    try:
        # NOTE: Sync action selection for now, maybe async later
        if hasattr(current_player, "select_action_async"):
            result = await current_player.select_action_async(state)
        else:
            result = current_player.select_action(state)

        action = result.action

        new_state = game.next_state(state, action)
        game_session["state"] = new_state
        
        # Log predicted scores if verbose logging is enabled
        if VERBOSE_AI_LOGGING and "current_player_mean_values" in result.info:
            mean_values = result.info["current_player_mean_values"]
            legal_actions = result.info.get("legal_actions", [])
            visit_counts = result.info.get("legal_action_visit_counts", [])
            # Find index of chosen action
            action_idx = legal_actions.index(action) if action in legal_actions else -1
            chosen_value = mean_values[action_idx] if action_idx >= 0 else None
            logger.info(
                "AI Player %d predicted scores: action=%s value=%.3f, best_value=%.3f, visit_counts=%s",
                current_player_id, action, chosen_value or 0, max(mean_values) if len(mean_values) > 0 else 0,
                visit_counts.tolist() if hasattr(visit_counts, 'tolist') else visit_counts
            )
        
        logger.info("AI move made for player %d. Action: %s", current_player_id, action)
        return {"success": True}
    except Exception as e:
        logger.error("Error making AI move: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error making AI move: {e}")


@app.get("/{game_type}/{game_id}", response_class=HTMLResponse)
async def serve_game_page(request: Request, game_type: str, game_id: int) -> HTMLResponse:
    logger.debug("Serving game page. Type: %s, ID: %d", game_type, game_id)

    # Basic validation
    # if game_type not in list_games(): ... # Relaxed for now to allow serving page even if registry mismatch (e.g. debugging)

    if game_id not in games:
        logger.error("Game not found. ID: %d", game_id)
        raise HTTPException(status_code=404, detail="Game not found")

    template_name = f"{game_type}.html"
    return templates.TemplateResponse(
        request=request, name=template_name, context={"request": request, "game_type": game_type, "game_id": game_id}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
