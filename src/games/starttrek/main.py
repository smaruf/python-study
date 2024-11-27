import json
import os
import time

def load_json_data(filepath):
    """Load JSON data from a file."""
    with open(filepath, 'r') as file:
        return json.load(file)

def save_json_data(filepath, data):
    """Save JSON data to a file."""
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def get_player_profile(players_db, username):
    """Retrieve player profile from the database."""
    return players_db['players'].get(username)

def update_player_profile(players_db, username, points, mission):
    """Update the player profile with new points and completed mission."""
    if username not in players_db['players']:
        players_db['players'][username] = {"username": username, "points": 0, "completed_missions": []}
    player = players_db['players'][username]
    player['points'] += points
    player['completed_missions'].append(mission)
    save_json_data('players.json', players_db)

def play_story(story, difficulty=1):
    """Play through the story and return the total points earned."""
    points = 0
    start_time = time.time()
    print(story['opening'])
    for planet in story['planets']:
        print(f"\nArrived at {planet['name']}: {planet['description']}")
        for action in planet['actions']:
            print(f"Do you want to {action['command']}? (yes/no)")
            if input().strip().lower() == 'yes':
                earned_points = action.get('points', 0) * difficulty
                points += earned_points
                print(f"{action['outcome']} (+{earned_points} points)")
    elapsed_time = time.time() - start_time
    print(f"\nMission completed in {elapsed_time:.2f} seconds with {points} points.")
    return points

def select_story():
    """Allow the user to select a story to play."""
    story_files = [f for f in os.listdir('.') if f.startswith('story_') and f.endswith('.json')]
    if not story_files:
        print("No story files found.")
        exit(1)
    print("Available Missions:")
    for idx, filename in enumerate(story_files):
        print(f"{idx + 1}: {filename[:-5]}")  # Remove '.json' from display
    while True:
        try:
            story_index = int(input("Choose a mission: ")) - 1
            if 0 <= story_index < len(story_files):
                break
            else:
                print("Invalid choice. Please choose a valid mission number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    return load_json_data(story_files[story_index])

def main():
    """Main function to run the game."""
    players_db = load_json_data('players.json')
    username = input("Enter your username: ").strip()
    while True:
        try:
            difficulty = int(input("Enter difficulty level (1-3): "))
            if 1 <= difficulty <= 3:
                break
            else:
                print("Please enter a difficulty level between 1 and 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    story = select_story()
    points = play_story(story, difficulty)
    update_player_profile(players_db, username, points, story['title'])

if __name__ == "__main__":
    main()
