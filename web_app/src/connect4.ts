import {
  BaseGameData,
  updateGameState,
  makeMove,
  startNewGame,
  currentPlayerType,
  GameRenderer,
} from './game_common.js'

interface Connect4GameData extends BaseGameData {
  rows: number
  columns: number
  state: number[][]
}

let previousBoard: number[][]
let gameOptions: { [key: string]: any }
let playerOptions: { [key: number]: { player_type: string;[key: string]: any } }

document.addEventListener('DOMContentLoaded', () => {
  console.log('connect4.ts loaded and DOMContentLoaded event fired.')

  const renderGame: GameRenderer<Connect4GameData> = (data) => {
    console.log('Rendering game with data:', data)
    const gameArea = document.getElementById('game')!
    const status = document.getElementById('status')!
    const modalBody = document.getElementById('modalBody')!
    const gameModalElement = document.getElementById('gameModal')

    // Check if bootstrap is available globally
    let gameModal: any = null;
    if (window.bootstrap && gameModalElement) {
      gameModal = new window.bootstrap.Modal(
        gameModalElement,
        {
          keyboard: false,
        },
      )
    }

    const newGameButton = document.getElementById('newGameButton')

    gameOptions = data.game_options
    playerOptions = data.player_options
    console.log('Updated gameOptions:', gameOptions)
    console.log('Updated playerOptions:', playerOptions)

    // Clear previous game board
    gameArea.innerHTML = ''

    // Create game board grid
    const grid = document.createElement('div')
    grid.classList.add('grid-container')

    // for (let row = data.rows - 1; row >= 0; row--) {
    for (let row = 0; row < data.rows; row++) {
      for (let col = 0; col < data.columns; col++) {
        const cell = document.createElement('div')
        cell.classList.add('grid-cell')
        cell.dataset.column = col.toString()
        cell.dataset.row = row.toString()

        if (data.state[row][col] === 1) {
          cell.classList.add('player1')
        } else if (data.state[row][col] === 2) {
          cell.classList.add('player2')
        }

        if (!data.is_terminal && currentPlayerType(data) === 'human') {
          cell.addEventListener('click', () => {
            console.log(`Cell clicked: Column ${col}`)
            makeMove<Connect4GameData>({ column: col + 1 }, renderGame)
          })
        }

        grid.appendChild(cell)
      }
    }

    gameArea.appendChild(grid)

    // Compare the new board with the previous board to detect the last move
    if (data.state) {
      findLastMove(data.state)
      previousBoard = data.state // Update the previous board state
    }

    if (data.is_terminal) {
      console.log('Game is terminal. Winner:', data.winner)
      const message = data.winner
        ? `🎉 <strong>Player ${data.winner} Wins!</strong> 🎉`
        : '🤝 <strong>The game is a draw!</strong> 🤝'

      modalBody.innerHTML = message
      if (gameModal) gameModal.show() // Show the Bootstrap modal when game ends

      if (newGameButton) {
        newGameButton.onclick = () => {
          startNewGame('connect4', gameOptions, playerOptions, renderGame)
            .catch((error) => console.error('Error starting new game:', error))
        }
      }
    } else {
      console.log('Game continuing. Current player:', data.current_player)
      status.textContent = `Current Turn: Player ${data.current_player}`
    }

    console.log('Finished rendering game')
  }

  function findLastMove(newBoard: number[][]) {
    if (!previousBoard) return // No previous board to compare to

    // Loop through the new board to find the difference
    for (let row = 0; row < newBoard.length; row++) {
      for (let col = 0; col < newBoard[row].length; col++) {
        // Check if there's a new piece (difference between old and new board)
        if (
          newBoard[row][col] != 0 &&
          newBoard[row][col] !== previousBoard[row][col]
        ) {
          highlightLastMove(row, col) // Highlight the last move
          return
        }
      }
    }
  }

  function highlightLastMove(row: number, column: number) {
    // Remove the 'last-move' class from all previous moves
    document.querySelectorAll('.grid-cell.last-move').forEach((cell) => {
      cell.classList.remove('last-move')
    })

    // Select the cell based on the row and column directly
    const lastMove = document.querySelector(
      `.grid-cell[data-column='${column}'][data-row='${row}']`,
    )

    // Add the 'last-move' class to the selected cell
    if (lastMove) {
      lastMove.classList.add('last-move')
    }
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
})

// Add global declaration for bootstrap
declare global {
  interface Window {
    bootstrap: any;
  }
}

