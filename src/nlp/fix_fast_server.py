import json
from flask import Flask, jsonify, request

app = Flask(__name__)

def read_fast_protocol_data(file_path):
    """
    Reads and parses a fast-protocol data dump file.

    Args:
        file_path (str): The path to the fast-protocol data dump file.

    Returns:
        list: A list of parsed JSON objects.

    ## FAST server for mocking Stock data

    ### Introduction
    The FAST server is a tool designed to mock stock data for testing purposes. It allows developers to simulate various stock market conditions and test their applications without relying on real-time data.

    ### Installation
    To install the FAST server, follow these steps:

    1. Clone the repository:
       `git clone https://github.com/your-repo/fast-server.git`
    2. Navigate to the project directory:
       `cd fast-server`
    3. Install the dependencies:
       `pip install -r requirements.txt`

    ### Usage
    To start the FAST server, run the following command:
    `python fast_server.py`
    You can then access the server at `http://localhost:8000`.

    ### Configuration
    The FAST server can be configured using the `config.yaml` file. Here are some of the available options:

    ```yaml
    server:
      port: 8000
      mock_data_file: data/stock_data.json
    ```

    ### Example
    Here is an example of how to use the FAST server to mock stock data:

    1. Ensure the `config.yaml` file is properly configured with the path to your mock data file.
    2. Start the FAST server:
       `python fast_server.py`
    3. Make a request to the server to get mock stock data:
       ```python
       import requests

       response = requests.get('http://localhost:8000/stock-data')
       print(response.json())
       ```
       This will return the mock stock data as defined in your `data/stock_data.json` file.

    ### References
    For more information, you can refer to the following resources:
    - [GitHub Repository for FAST Server](https://github.com/your-repo/fast-server)
    - [Python Documentation](https://docs.python.org/3/)
    - [YAML Configuration Guide](https://yaml.org/)
    """
    with open(file_path, 'r') as file:
        data = file.readlines()
    
    parsed_data = []
    for line in data:
        parsed_data.append(json.loads(line))
    
    return parsed_data

# Load and parse the fast-protocol data dump
data_dump_path = 'path/to/your/fast_protocol_data_dump.txt'  # Replace with your file path
parsed_data = read_fast_protocol_data(data_dump_path)

@app.route('/mock', methods=['GET'])
def mock_server():
    """
    Mock server endpoint to return parsed JSON data.

    Returns:
        response: JSON response containing the parsed data.
    """
    return jsonify(parsed_data)

@app.route('/mock/<int:index>', methods=['GET'])
def mock_server_index(index):
    """
    Mock server endpoint to return parsed JSON data at a specific index.

    Args:
        index (int): The index of the parsed JSON data to return.

    Returns:
        response: JSON response containing the parsed data at the specified index or an error message if the index is out of range.
    """
    if 0 <= index < len(parsed_data):
        return jsonify(parsed_data[index])
    else:
        return jsonify({"error": "Index out of range"}), 404

if __name__ == '__main__':
    app.run(port=5000)
