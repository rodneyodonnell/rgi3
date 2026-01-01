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
  board: number[][]
  legal_actions: [number, number][]
  game_options: { [key: string]: any }
  player_options: { [key: number]: { player_type: string; [key: string]: any } }
}

let gameOptions: { [key: string]: any } = {}
let playerOptions: { [key: number]: { player_type: string; [key: string]: any } } = {}

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
        const human_row = data.rows - row
        const human_col = col + 1

        const cell = document.createElement('div')
        cell.classList.add('grid-cell')
        cell.dataset.row = human_row.toString()
        cell.dataset.col = human_col.toString()

        console.log(
          `Creating cell at (${human_row}, ${human_col}) with state: ${data.board[row][col]}`,
        )

        // Set cell color based on state
        if (data.board[row][col] === 1) {
          cell.classList.add('player1')
        } else if (data.board[row][col] === 2) {
          cell.classList.add('player2')
        }

        // Highlight legal moves
        if (data.legal_actions.some(([r, c]) => r === human_row && c === human_col)) {
          cell.classList.add('legal-move')
          console.log(`Legal move at (${human_row}, ${human_col})`)
        }

        if (!data.is_terminal && currentPlayerType(data) === 'human') {
          cell.addEventListener('click', () => {
            console.log('Cell clicked:', { row: human_row, col: human_col })
            makeMove<OthelloGameData>({ row: human_row, col: human_col }, renderGame)
          })
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
      status.textContent = `Current Turn: Player ${data.current_player}`
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