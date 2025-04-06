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
