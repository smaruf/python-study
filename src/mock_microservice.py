import responses
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

# Setup mock responses for external APIs
@responses.activate
def setup_mock_responses():
    responses.add(
        responses.GET,
        "http://localhost:8000/api/data",
        json={"data": "Dynamic data based on type"},
        status=200
    )

    responses.add(
        responses.GET,
        "http://localhost:8000/api/status",
        json={"status": "Dynamic status based on id"},
        status=200
    )

    # Example calls to external API with params
    data = requests.get("http://localhost:8000/api/data", params={"type": "example"}).json()
    status = requests.get("http://localhost:8000/api/status", params={"id": "123"}).json()

    return {
        "data_response": data,
        "status_response": status
    }

# Flask route for /api/data with parameter 'type'
@app.route('/api/data', methods=['GET'])
def data():
    data_type = request.args.get('type', 'default')
    # Will return different data based on what 'type' is requested
    return jsonify({"data": f"Mocked data response for type {data_type}"}), 200

# Flask route for /api/status with parameter 'id'
@app.route('/api/status', methods=['GET'])
def status():
    status_id = request.args.get('id', 'none')
    # Will return different status based on what 'id' is provided
    return jsonify({"status": f"Mocked status response for ID {status_id}"}), 200

def run_app():
    # Setup mock before running the server
    setup_mock_responses()
    # Run Flask app to bind to port 8000
    app.run(port=8000, debug=True)

if __name__ == "__main__":
    run_app()
