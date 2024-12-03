import json
import random

def load_player_config():
    with open('player_config.json', 'r') as file:
        config = json.load(file)
    return config

def ai_action():
    # Simple AI: just returns a random action
    return random.choice(['scan', 'contact', 'explore'])

def perform_action(player, action):
    if player['type'] == 'AI':
        action = ai_action()
    print(f"{player['name']} chooses to {action}")

def main():
    config = load_player_config()
    
    if config['mode'] == 'offline':
        print("Starting the game in offline mode...")
        for player in config['players']:
            action = input(f"{player['name']} - Enter your action (scan, contact, explore): ") if player['type'] == 'human' else ai_action()
            perform_action(player, action)

    else:
        print("Online mode is currently not implemented.")
        # Implement client-server communication for online gameplay

if __name__ == '__main__':
    main()
