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
