package main

import (
    "bufio"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "os"
    "strconv"
    "strings"
    "time"
)

type Action struct {
    Command string `json:"command"`
    Outcome string `json:"outcome"`
    Points  int    `json:"points"`
}

type Planet struct {
    Name        string   `json:"name"`
    Description string   `json:"description"`
    Actions     []Action `json:"actions"`
}

type Story struct {
    Title   string   `json:"title"`
    Opening string   `json:"opening"`
    Planets []Planet `json:"planets"`
    End     string   `json:"end"`
}

type Player struct {
    Username         string   `json:"username"`
    Points           int      `json:"points"`
    CompletedMissions []string `json:"completed_missions"`
}

type PlayersDB struct {
    Players map[string]Player `json:"players"`
}

func main() {
    reader := bufio.NewReader(os.Stdin)

    // Get username
    fmt.Println("Enter your username:")
    username := getInput(reader)

    // Load player database
    playersDB := loadPlayers("players.json")
    player, exists := playersDB.Players[username]

    if !exists {
        player = Player{Username: username, Points: 0, CompletedMissions: []string{}}
        fmt.Println("New player detected. Welcome!")
    }

    // Get difficulty level
    fmt.Println("Enter difficulty level (1, 2, or 3):")
    difficulty := getDifficulty(reader)

    // Load story
    story := loadStory("story_mission1_vulcan.json")
    fmt.Println(story.Opening)

    // Start mission
    totalPoints := 0
    startTime := time.Now()

    for _, planet := range story.Planets {
        fmt.Printf("\nArrived at %s: %s\n", planet.Name, planet.Description)
        for _, action := range planet.Actions {
            fmt.Printf("Do you want to %s? (yes/no)\n", action.Command)
            if getInput(reader) == "yes" {
                totalPoints += action.Points * difficulty
                fmt.Println(action.Outcome + fmt.Sprintf(" (+%d points)", action.Points*difficulty))
            }
        }
    }

    // Mission completion
    elapsedTime := time.Since(startTime)
    fmt.Printf("\nMission completed in %.2f seconds with %d points.\n", elapsedTime.Seconds(), totalPoints)

    player.Points += totalPoints
    player.CompletedMissions = append(player.CompletedMissions, story.Title)
    playersDB.Players[username] = player
    savePlayers("players.json", playersDB)

    fmt.Println(story.End)
}

// getInput reads input from the user and trims whitespace
func getInput(reader *bufio.Reader) string {
    input, _ := reader.ReadString('\n')
    return strings.TrimSpace(input)
}

// getDifficulty reads and validates the difficulty level from the user
func getDifficulty(reader *bufio.Reader) int {
    for {
        difficultyStr := getInput(reader)
        difficulty, err := strconv.Atoi(difficultyStr)
        if err == nil && difficulty >= 1 && difficulty <= 3 {
            return difficulty
        }
        fmt.Println("Invalid difficulty level. Please enter 1, 2, or 3:")
    }
}

// loadPlayers loads the player database from a JSON file
func loadPlayers(filePath string) PlayersDB {
    file, err := ioutil.ReadFile(filePath)
    if err != nil {
        fmt.Printf("Error reading players file: %v\n", err)
        os.Exit(1)
    }
    var playersDB PlayersDB
    if err := json.Unmarshal(file, &playersDB); err != nil {
        fmt.Printf("Error unmarshalling players data: %v\n", err)
        os.Exit(1)
    }
    return playersDB
}

// savePlayers saves the player database to a JSON file
func savePlayers(filePath string, playersDB PlayersDB) {
    data, err := json.MarshalIndent(playersDB, "", "  ")
    if err != nil {
        fmt.Printf("Error marshalling players data: %v\n", err)
        os.Exit(1)
    }
    if err := ioutil.WriteFile(filePath, data, 0644); err != nil {
        fmt.Printf("Error writing players file: %v\n", err)
        os.Exit(1)
    }
}

// loadStory loads a story from a JSON file
func loadStory(filePath string) Story {
    file, err := ioutil.ReadFile(filePath)
    if err != nil {
        fmt.Printf("Error reading story file: %v\n", err)
        os.Exit(1)
    }
    var story Story
    if err := json.Unmarshal(file, &story); err != nil {
        fmt.Printf("Error unmarshalling story data: %v\n", err)
        os.Exit(1)
    }
    return story
}
