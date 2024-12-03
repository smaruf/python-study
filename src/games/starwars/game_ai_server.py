from flask import Flask, request, jsonify, session
from uuid import uuid4
import openai
import torch
import torch.nn as nn
import random

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Game state
game_state = {'players': {}}

# Load and set up Neural Network (Mock)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(10, 5)  # Placeholder architecture
    
    def forward(self, x):
        return self.fc(x)

model = SimpleNN()
model.eval()

# Neural Network Predict Function (Mock)
def predict(input_data):
    with torch.no_grad():
        tensor_data = torch.FloatTensor([input_data])
        output = model(tensor_data)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Simple Random AI Logic
def random_action():
    actions = ['advance', 'retreat', 'collect resource', 'build defense']
    return random.choice(actions)

# OpenAI's ChatGPT Interaction
def interact_with_chatgpt(prompt):
    try:
        response = openai.Completion.create(
          engine="text-davinci-002",
          prompt=prompt,
          max_tokens=100
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print("Failed to contact ChatGPT:", e)
        return None

@app.route('/action', methods=['POST'])
def post_action():
    player_id = session.get('player_id')
    if not player_id or player_id not in game_state['players']:
        return jsonify({'error': 'Player not recognized or not in session'}), 403
    
    action = request.json.get('action', '')
    prompt = f"How should the game respond to the action: {action}?"

    # Try using ChatGPT
    response = interact_with_chatgpt(prompt)
    if response is None:
        print("ChatGPT unavailable, falling back to custom model.")
        response = predict([random.random() for _ in range(10)])  # Example input
        if response is None:
            print("Custom model failed, using random action.")
            response = random_action()
    
    game_state['players'][player_id]['actions'].append(action)
    return jsonify({'AI Response': response}), 200

@app.route('/join', methods=['POST'])
def join_game():
    player_name = request.json.get('name')
    player_id = str(uuid4())
    game_state['players'][player_id] = {
        'name': player_name,
        'actions': []
    }
    session['player_id'] = player_id
    return jsonify({'message': f'{player_name} joined the game.', 'player_id': player_id}), 200

if __name__ == '__main__':
    app.run(debug=True)
