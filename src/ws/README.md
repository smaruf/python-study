### WebSocket (ws) Learning:

The `ws` folder contains scripts and configurations for learning and implementing WebSocket communication. Below is an overview of the contents and their purposes:

#### Contents

- `README.md`: Provides an overview and resources for learning WebSocket communication.
- `main.js`: Initializes the user interface for a Connect Four game.
- `run_interactive.txt`: Instructions for connecting to a WebSocket server interactively.
- `app_ws.py`: The WebSocket server script handles and prints incoming messages.
- `index_ws.html`: HTML file for the Connect Four game, integrating the `main.js` script.
- `server_ws.py`: WebSocket server script that echoes received messages.
- `client_ws.py`: WebSocket client script that sends and receives messages.
  
#### Instructions

##### WebSocket Server

To run the WebSocket server, use the following command:

```sh
python app_ws.py
```

This will start a WebSocket server on port 8001.

##### WebSocket Client

To run the WebSocket client, use the following command:

```sh
python client_ws.py
```

This will connect to the WebSocket server on `ws://localhost:8765` and send a "Hello world!" message.

##### Connect Four Game

To run the Connect Four game, open the `index_ws.html` file in a web browser. The game board will be initialized, and you can interact with it.

#### Summary

The scripts in the `ws` folder provide a starting point for learning and implementing WebSocket communication. Follow the instructions for running the server and client scripts, and explore the Connect Four game for a practical example of WebSocket usage.

#### Additional Resources

For more information on WebSockets, refer to the following resources:

- [WebSocket Client Documentation](https://websocket-client.readthedocs.io/en/latest/)
- [WebSockets Documentation](https://websockets.readthedocs.io/en/stable/)
