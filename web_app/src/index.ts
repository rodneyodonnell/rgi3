// web_app/static/index.js

// Fetch and populate model dropdowns
async function loadModels() {
  try {
    const response = await fetch('/api/models')
    const data = await response.json()
    const models: string[] = data.models || []

    // Populate all model select elements
    document.querySelectorAll('.model-select').forEach((select) => {
      const selectEl = select as HTMLSelectElement
      // Keep the first "-- Select Model --" option
      while (selectEl.options.length > 1) {
        selectEl.remove(1)
      }
      models.forEach((modelPath) => {
        const option = document.createElement('option')
        option.value = modelPath
        // Show just the experiment name and generation for readability
        const parts = modelPath.split('/')
        const expName = parts[1] || ''
        const genFile = parts[parts.length - 1] || ''
        option.text = `${expName} / ${genFile}`
        selectEl.add(option)
      })
    })
  } catch (error) {
    console.error('Failed to load models:', error)
  }
}

// Show/hide AI options based on player type selection
function setupPlayerTypeListeners() {
  document.querySelectorAll('.player-select').forEach((select) => {
    const selectEl = select as HTMLSelectElement
    selectEl.addEventListener('change', () => {
      const game = selectEl.dataset.game
      const player = selectEl.dataset.player
      const optionsDiv = document.getElementById(`${game}Player${player}Options`)
      const playerType = selectEl.value

      // Show AI options for alphazero variants that need configuration
      if (playerType === 'alphazero_custom' || playerType === 'zerozero') {
        optionsDiv?.style.setProperty('display', 'block')
      } else {
        optionsDiv?.style.setProperty('display', 'none')
      }
    })
  })
}

// Setup simulation slider value display
function setupSimulationSliders() {
  document.querySelectorAll('input[type="range"]').forEach((slider) => {
    const sliderEl = slider as HTMLInputElement
    const valueSpan = document.getElementById(`${sliderEl.id}Value`)
    if (valueSpan) {
      sliderEl.addEventListener('input', () => {
        valueSpan.textContent = sliderEl.value
      })
    }
  })
}

// Get player options from form
function getPlayerOptions(game: string, playerNum: number): { player_type: string;[key: string]: any } {
  const selectEl = document.getElementById(`${game}Player${playerNum}`) as HTMLSelectElement
  const playerType = selectEl.value

  const options: { player_type: string;[key: string]: any } = {
    player_type: playerType,
  }

  // Add AI-specific options if applicable
  if (playerType === 'alphazero_custom' || playerType === 'zerozero') {
    const simsSlider = document.getElementById(`${game}Player${playerNum}Sims`) as HTMLInputElement
    if (simsSlider) {
      options.simulations = parseInt(simsSlider.value, 10)
    }

    if (playerType === 'alphazero_custom') {
      const modelSelect = document.getElementById(`${game}Player${playerNum}Model`) as HTMLSelectElement
      if (modelSelect && modelSelect.value) {
        options.model_path = modelSelect.value
      }
    }
  }

  return options
}

document.addEventListener('DOMContentLoaded', () => {
  // Load models for dropdowns
  loadModels()

  // Setup event listeners
  setupPlayerTypeListeners()
  setupSimulationSliders()

  const connect4Form = document.getElementById('connect4Form')
  const othelloForm = document.getElementById('othelloForm')

  connect4Form?.addEventListener('submit', (e) => {
    e.preventDefault()
    const gameOptions = {}
    const playerOptions = {
      1: getPlayerOptions('connect4', 1),
      2: getPlayerOptions('connect4', 2),
    }
    startGame('connect4', gameOptions, playerOptions)
  })

  othelloForm?.addEventListener('submit', (e) => {
    e.preventDefault()
    const gameOptions = {}
    const playerOptions = {
      1: getPlayerOptions('othello', 1),
      2: getPlayerOptions('othello', 2),
    }
    startGame('othello', gameOptions, playerOptions)
  })
})

function startGame(
  gameType: string,
  gameOptions: { [key: string]: any },
  playerOptions: { [key: number]: { player_type: string;[key: string]: any } }
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
        return response.json().then((data) => {
          throw new Error(data.detail || `HTTP error! Status: ${response.status}`)
        })
      }
      return response.json()
    })
    .then((data) => {
      console.log('Game created with ID:', data.game_id)
      window.location.href = `/${gameType}/${data.game_id}${window.location.search}`
    })
    .catch((error) => {
      console.error('Error creating game:', error)
      alert(`Failed to create game: ${error.message}`)
    })
}
