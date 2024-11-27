# Star Trek Text-Based Game

This is a text-based **Star Trek** game where players can embark on various missions, earn points, and track their progress.

## How to Play

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/smaruf/python-study.git
   cd python-study/src/games/starttrek
   ```

1. **Run the Game:**
    ```bash
    python main.py
    ```

1. **Gameplay Instruction:**
   - Enter your username when prompted.
   - Select a difficulty level (1 to 3).
   - Choose a mission from the list of available stories.
   - Follow the prompts to make decisions during your mission.
   - Earn points based on your actions and the chosen difficulty level.
   - Your progress will be saved in `players.json`.
  
#### Example Commands
```
# Starting the game
  python main.py

# Example interaction
# Enter your username: captainkirk
# Enter difficulty level (1-3): 2
# Choose a mission: 1
```

### Files and Directories
- **main.py**: The main script to start the game.
- **players.json**: A file where player profiles and progress are stored.
- **story_<name>.json**: Files containing different missions and stories for the game.

## Adding New Missions
To add a new mission, create a new JSON file in the same directory with the prefix `story_` and follow the structure of existing story files. Below is an example structure for a story file:

```json
{
  "title": "Mission Title",
  "opening": "Mission opening description...",
  "planets": [
    {
      "name": "Planet Name",
      "description": "Description of the planet...",
      "actions": [
        {
          "command": "action command",
          "outcome": "outcome of the action",
          "points": 10
        }
      ]
    }
  ]
}
```


### Contributing

Contributions to improve the game or add new features are welcome. Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License.
