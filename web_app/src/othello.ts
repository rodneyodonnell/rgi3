import {
  BaseGameData,
  updateGameState,
  makeMove,
  startNewGame,
  currentPlayerType,
  GameRenderer,
} from './game_common.js'

interface OthelloGameData extends BaseGameData {
  rows: number
  columns: number
  state: number[][]
  legal_actions: [number, number][]
  game_options: { [key: string]: any }
  player_options: { [key: number]: { player_type: string;[key: string]: any } }
}

let gameOptions: { [key: string]: any } = {}
let playerOptions: { [key: number]: { player_type: string;[key: string]: any } } = {}

document.addEventListener('DOMContentLoaded', () => {
  console.log('othello.ts loaded and DOMContentLoaded event fired.')

  // Now renderGame is explicitly typed with GameRenderer
  const renderGame: GameRenderer<OthelloGameData> = (data) => {
    console.log('Rendering game with data:', JSON.stringify(data, null, 2))
    const gameArea = document.getElementById('game')
    const status = document.getElementById('status')
    const modalBody = document.getElementById('modalBody')
    const gameModal = new window.bootstrap.Modal(
      document.getElementById('gameModal')!,
      {
        keyboard: false,
      },
    )
    const newGameButton = document.getElementById('newGameButton')

    if (!gameArea || !status) {
      console.error('Required DOM elements not found.')
      return
    }

    gameOptions = data.game_options
    playerOptions = data.player_options
    console.log('Updated gameOptions:', gameOptions)
    console.log('Updated playerOptions:', playerOptions)

    // Clear previous game board
    gameArea.innerHTML = ''

    // Create game board grid
    const grid = document.createElement('div')
    grid.classList.add('grid-container')
    grid.style.gridTemplateColumns = `repeat(${data.columns}, 1fr)`
    grid.style.gridTemplateRows = `repeat(${data.rows}, 1fr)`

    console.log(
      `Creating grid with ${data.rows} rows and ${data.columns} columns`,
    )

    for (let row = 0; row < data.rows; row++) {
      for (let col = 0; col < data.columns; col++) {
        // Backend coords (0,0) is Top-Left. 
        // Backend logic uses (row, col). Frontend display matches.

        // Legal actions might be in human coords or grid coords? 
        // OthelloGame._to_human_coords converts to 1-based.
        // Let's assume legal_actions in data are formatted as backend returns them.
        // Looking at main.py -> serialize_state -> recursive -> might preserve tuples or lists.
        // Othello legal_actions return list of tuples.

        const cell = document.createElement('div')
        cell.classList.add('grid-cell')
        cell.dataset.row = row.toString()
        cell.dataset.col = col.toString()

        // Set cell color based on state
        // 1 = Player 1 (Black), 2 = Player 2 (White)
        if (data.state[row][col] === 1) {
          cell.classList.add('player1')
        } else if (data.state[row][col] === 2) {
          cell.classList.add('player2')
        }

        // Highlight legal moves
        // Need to check how legal_actions are returned.
        // OthelloGame returns human coords (1-based)? 
        // Let's check legal_actions content from console logs or source.
        // OthelloGame.legal_actions calls _get_legal_moves -> adds _to_human_coords.
        // So legal actions are 1-based (row, col).
        // My loop is 0-based.
        const human_r = row + 1
        const human_c = col + 1

        // Check legal actions. data.legal_actions is likely [[r, c], [r, c]] (list of lists due to json)
        // or list of tuples if handled.
        // Safely check:
        const isLegal = data.legal_actions && data.legal_actions.some((move: any) => {
          // handle both array and object if serialization does something weird
          const r = Array.isArray(move) ? move[0] : move[0]
          const c = Array.isArray(move) ? move[1] : move[1]
          return r === human_r && c === human_c
        })

        if (isLegal) {
          cell.classList.add('legal-move')

          if (!data.is_terminal && currentPlayerType(data) === 'human') {
            cell.addEventListener('click', () => {
              console.log('Cell clicked:', { row: human_r, col: human_c })
              // Send human coords if that's what backend expects?
              // OthelloGame.make_move expects legal action. Legal actions are human coords?
              // OthelloGame.next_state checks "if action not in legal_actions".
              // So yes, we must send (human_row, human_col).
              // Backend make_move expects dict.

              // We need to send {row: ..., col: ...} or just the tuple?
              // main.py make_move attempts to parse.
              // "if 'column' in action_data" -> Connect4 specific logic in main.py?
              // I might need to update main.py to handle Othello moves (row, col) too!

              // Let's send {row: human_r, col: human_c} and update main.py if needed.
              makeMove({ row: human_r, col: human_c }, renderGame)
            })
          }
        }

        grid.appendChild(cell)
      }
    }

    gameArea.appendChild(grid)

    // Update game status
    if (data.is_terminal) {
      console.log('Game is terminal. Winner:', data.winner)
      const message = data.winner
        ? `🎉 <strong>Player ${data.winner} Wins!</strong> 🎉`
        : '🤝 <strong>The game is a draw!</strong> 🤝'

      modalBody!.innerHTML = message
      gameModal.show() // Show the Bootstrap modal when game ends

      newGameButton!.onclick = () => {
        startNewGame('othello', gameOptions, playerOptions, renderGame)
          .catch((error) => console.error('Error starting new game:', error))
      }
    } else {
      console.log('Game continuing. Current player:', data.current_player)
      status.textContent = `Current Turn: Player ${data.current_player} (${data.current_player === 1 ? 'Black' : 'White'})`
    }

    console.log('Finished rendering game')
  }

  // Initial game state fetch
  updateGameState(renderGame)
    .then((data) => {
      gameOptions = data.game_options
      playerOptions = data.player_options
    })
    .catch((error) =>
      console.error('Error fetching initial game state:', error),
    )

  // Set up new game button
  document.getElementById('newGameButton')?.addEventListener('click', () => {
    startNewGame('othello', gameOptions, playerOptions, renderGame)
  })
})