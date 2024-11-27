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
	fmt.Println("Enter your username:")
	username, _ := reader.ReadString('\n')
	username = strings.TrimSpace(username)

	playersDB := loadPlayers("players.json")
	player, exists := playersDB.Players[username]

	if !exists {
		player = Player{Username: username, Points: 0, CompletedMissions: []}
		fmt.Println("New player detected. Welcome!")
	}

	fmt.Println("Enter difficulty level (1, 2, or 3):")
	difficultyStr, _ := reader.ReadString('\n')
	difficulty, _ := strconv.Atoi(strings.TrimSpace(difficultyStr))

	story := loadStory("story_mission1_vulcan.json")
	fmt.Println(story.Opening)
	totalPoints := 0
	startTime := time.Now()

	for _, planet := range story.Planets {
		fmt.Printf("\nArrived at %s: %s\n", planet.Name, planet.Description)
		for _, action := range planet.Actions {
			fmt.Printf("Do you want to %s? (yes/no)\n", action.Command)
			response, _ := reader.ReadString('\n')
			if strings.TrimSpace(response) == "yes" {
				totalPoints += action.Points * difficulty
				fmt.Println(action.Outcome + fmt.Sprintf(" (+%d points)", action.Points*difficulty))
			}
		}
	}

	elapsedTime := time.Since(startTime)
	fmt.Printf("\nMission completed in %v seconds with %d points.\n", elapsedTime.Seconds(), totalPoints)

	player.Points += totalPoints
	player.CompletedMissions = append(player.CompletedMissions, story.Title)
	playersDB.Players[username] = player
	savePlayers("players.json", playersDB)

	fmt.Println(story.End)
}

func loadPlayers(filePath string) PlayersDB {
	file, err := ioutil.ReadFile(filePath)
	if err != nil {
		panic(err)
	}
	var playersDB PlayersDB
	json.Unmarshal(file, &playersDB)
	return playersDB
}

func savePlayers(filePath string, playersDB PlayersDB) {
	data, err := json.MarshalIndent(playersDB, "", "  ")
	if err != nil {
		panic(err)
	}
	err = ioutil.WriteFile(filePath, data, 0644)
	if err != nil {
		panic(err)
	}
}

func loadStory(filePath string) Story {
	file, err := ioutil.ReadFile(filePath)
	if err != nil {
		panic(err)
	}
	var story Story
	json.Unmarshal(file, &story)
	return story
}
