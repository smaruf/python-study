import threading
import time
import random
from queue import Queue, Empty
from contextlib import suppress

def player(name, table, game_end_event):
    while not game_end_event.is_set():
        try:
            ball = table.get(timeout=0.1)  # Try to get the ball
            print(f"{name} received the ball.")
            # Simulate random wait time between 0 to 10 seconds
            wait_time = random.randint(0, 10)
            time.sleep(wait_time)

            # Check if the game is still on before hitting back
            if not game_end_event.is_set():
                print(f"{name} hits the ball back after waiting for {wait_time} seconds.")
                table.put("ball")
            else:
                print(f"{name} has the ball. Game over!")
        except Empty:
            # No ball to receive right now, continue waiting
            continue

def main():
    # Create a queue to serve as the table
    table = Queue()

    # Event to signal when the game is over
    game_end_event = threading.Event()

    # Start players as threads
    threads = [
        threading.Thread(target=player, args=("Player 1", table, game_end_event)),
        threading.Thread(target=player, args=("Player 2", table, game_end_event)),
    ]

    for thread in threads:
        thread.start()

    # Start the game by putting the first ball on the table
    table.put("ball")

    # Let the game run for 2 minutes
    time.sleep(120)  # 120 seconds is 2 minutes

    # Signal that the game is over
    game_end_event.set()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("Ping pong game finished.")

if __name__ == "__main__":
    main()
