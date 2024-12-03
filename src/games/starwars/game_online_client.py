import requests

BASE_URL = "http://127.0.0.1:5000"

def join_game(name):
    response = requests.post(f"{BASE_URL}/join", json={'name': name})
    return response.json()

def send_action(player_id, action):
    session = requests.Session()
    session.cookies.set('session', player_id)
    response = session.post(f"{BASE_URL}/action", json={'action': action})
    return response.json()

def get_game_state():
    response = requests.get(f"{BASE_URL}/game_state")
    return response.json()

if __name__ == "__main__":
    player_name = input("Enter your name: ")
    result = join_game(player_name)
    player_id = result.get('player_id')

    while True:
        action = input("Enter an action (or 'exit' to quit): ")
        if action.lower() == 'exit':
            break
        result = send_action(player_id, action)
        print("Server response:", result)

    game_state = get_game_state()
    print("Current Game State:", game_state)
