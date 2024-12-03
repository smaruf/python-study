from flask import Flask, request, jsonify, session
from uuid import uuid4

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secret key in production

# Synchronize Game state across different players
game_state = {
    'players': {},
    # You can add more game-specific state information here
}

@app.route('/join', methods=['POST'])
def join_game():
    player_name = request.json.get('name')
    if not player_name:
        return jsonify({'error': 'Name is required'}), 400

    player_id = str(uuid4())
    game_state['players'][player_id] = {
        'name': player_name,
        'actions': []
    }

    session['player_id'] = player_id
    return jsonify({'message': f'{player_name} joined the game.', 'player_id': player_id}), 200

@app.route('/action', methods=['POST'])
def post_action():
    action = request.json.get('action')
    player_id = session.get('player_id')
    
    if not player_id or player_id not in game_state['players']:
        return jsonify({'error': 'Player not recognized or not in session'}), 403
    
    if not action:
        return jsonify({'error': 'Action is required'}), 400

    # Record the action - simplistic handling
    game_state['players'][player_id]['actions'].append(action)
    # Return the updated game state (or part of it) to the player
    return jsonify({'message': 'Action recorded', 'game_state': game_state['players'][player_id]}), 200

@app.route('/game_state', methods=['GET'])
def get_game_state():
    return jsonify(game_state), 200

if __name__ == '__main__':
    app.run(debug=True)
