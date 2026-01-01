// web_app/static/index.js

document.addEventListener('DOMContentLoaded', () => {
  const connect4Form = document.getElementById('connect4Form')
  const othelloForm = document.getElementById('othelloForm')

  connect4Form?.addEventListener('submit', (e) => {
    e.preventDefault()
    const gameOptions = {
      // Add any Connect4-specific game options here
    }
    const playerOptions = {
      1: {
        player_type: (document.getElementById('connect4Player1') as HTMLSelectElement).value,
      },
      2: {
        player_type: (document.getElementById('connect4Player2') as HTMLSelectElement).value,
      },
    }
    startGame('connect4', gameOptions, playerOptions)
  })

  othelloForm?.addEventListener('submit', (e) => {
    e.preventDefault()
    const gameOptions = {
      // Add any Othello-specific game options here
    }
    const playerOptions = {
      1: {
        player_type: (document.getElementById('othelloPlayer1') as HTMLSelectElement).value,
      },
      2: {
        player_type: (document.getElementById('othelloPlayer2') as HTMLSelectElement).value,
      },
    }
    startGame('othello', gameOptions, playerOptions)
  })
})

function startGame(
  gameType: string,
  gameOptions: { [key: string]: any },
  playerOptions: { [key: number]: { player_type: string; [key: string]: any } }
) {
  console.log('Starting game:', { gameType, gameOptions, playerOptions })
  fetch('/games/new', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      game_type: gameType,
      game_options: gameOptions,
      player_options: playerOptions,
    }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`)
      }
      return response.json()
    })
    .then((data) => {
      console.log('Game created with ID:', data.game_id)
      window.location.href = `/${gameType}/${data.game_id}${window.location.search}`
    })
    .catch((error) => {
      console.error('Error creating game:', error)
      alert('Failed to create game. Please try again.')
    })
}
