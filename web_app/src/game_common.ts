
const urlParams = new URLSearchParams(window.location.search)
const aiIntervalMs = Number(urlParams.get('ai_interval_ms') ?? '150')

declare global {
  interface Window {
    bootstrap: any
  }
}


export interface BaseGameData {
  is_terminal: boolean
  winner: number | null
  current_player: number
  game_options: { [key: string]: any }
  player_options: { [key: number]: { player_type: string;[key: string]: any } }
}


export function getCurrentGameId(): string {
  return window.location.pathname.split('/').pop() || ''
}

export function showErrorToast(message: string) {
  const toastBody = document.getElementById('toastBody')
  if (!toastBody) {
    console.error('Toast body element not found.')
    return
  }
  toastBody.textContent = message

  const errorToastElement = document.getElementById('errorToast')
  if (!errorToastElement) {
    console.error('Error toast element not found.')
    return
  }

  const errorToast = new window.bootstrap.Toast(errorToastElement)
  errorToast.show()
  console.log(`Error toast displayed: ${message}`)
}

export type GameRenderer<T extends BaseGameData> = (data: T) => void

export function updateGameState<T extends BaseGameData>(
  renderGame: GameRenderer<T>
): Promise<T> {
  const gameId = getCurrentGameId()
  console.log('Updating game state for game ID:', gameId)

  return fetch(`/games/${gameId}/state`)
    .then((response) => {
      if (!response.ok) {
        throw new Error('Failed to fetch game state')
      }
      return response.json()
    })
    .then((data) => {
      console.log('Received game state:', data)
      renderGame(data as T)

      // Check if it's AI's turn after updating the state
      if (!data.is_terminal && currentPlayerType(data as BaseGameData) !== 'human') {
        makeAIMove(renderGame)
      }

      return data as T
    })
    .catch((error) => {
      console.error('Error fetching game state:', error)
      showErrorToast('Failed to fetch game state.')
      throw error
    })
}

export function makeMove<T extends BaseGameData>(
  action: { [key: string]: number | string },
  renderGame: GameRenderer<T>
) {
  const gameId = getCurrentGameId()
  console.log(`Making move for game ${gameId}:`, action)
  fetch(`/games/${gameId}/move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(action),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log('Move response:', data)
      if (data.success) {
        updateGameState(renderGame)
      } else {
        showErrorToast(data.error || 'Unknown error occurred')
      }
    })
    .catch((error) => {
      console.error('Error making move:', error)
      showErrorToast(error.message || 'Failed to make move.')
    })
}

export function startNewGame(
  gameType: string,
  gameOptions: { [key: string]: any },
  playerOptions: { [key: number]: { player_type: string;[key: string]: any } },
  renderGame: GameRenderer<any>
): Promise<void> {
  console.log(`Starting a new ${gameType} game with options:`, { gameOptions, playerOptions })

  return fetch('/games/new', {
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
        throw new Error(
          `Failed to create a new game. Status code: ${response.status}`,
        )
      }
      return response.json()
    })
    .then((data) => {
      console.log('New game created with ID:', data.game_id)
      // Update the URL with the new game ID
      window.history.pushState(
        {},
        '',
        `/${gameType}/${data.game_id}${window.location.search}`,
      )

      const gameModal = window.bootstrap.Modal.getInstance(
        document.getElementById('gameModal')!,
      )
      if (gameModal) {
        gameModal.hide()
      }

      return updateGameState(renderGame)
    })
    .catch((error) => {
      console.error('Error creating new game:', error)
      showErrorToast('Failed to create a new game. Please try again.')
      throw error
    })
}

export function makeAIMove(renderGame: GameRenderer<any>) {
  const gameId = getCurrentGameId()
  console.log('Attempting AI move for game ID:', gameId)

  const startTime = Date.now()

  fetch(`/games/${gameId}/ai_move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error('Failed to make AI move')
      }
      return response.json()
    })
    .then((data) => {
      console.log('AI move response:', data)
      if (data.success) {
        const elapsedTime = Date.now() - startTime
        const remainingDelay = Math.max(0, aiIntervalMs - elapsedTime)

        // Add artificial delay so AI doesn't feel too "fast".
        setTimeout(() => {
          updateGameState(renderGame)
        }, remainingDelay)
      }
    })
    .catch((error) => {
      console.error('Error making AI move:', error)
    })
}

export function currentPlayerType(data: BaseGameData): string {
  const playerOptions = data.player_options[data.current_player] || { player_type: 'human' }
  return playerOptions.player_type
}
