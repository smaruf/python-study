import httpretty
import requests
import json

# Register httpretty to intercept requests for the specified URIs
def setup_mock_api_endpoints():
    httpretty.enable()
    httpretty.register_uri(
        httpretty.GET,
        "http://example.com/api1",
        body=json.dumps({"message": "Response from API 1"}),
        content_type="application/json",
        status=200
    )

    httpretty.register_uri(
        httpretty.GET,
        "http://example.com/api2",
        body=json.dumps({"info": "Details from API 2"}),
        content_type="application/json",
        status=200
    )

def fetch_data_from_apis():
    # Make HTTP GET requests to the mocked APIs
    response_api1 = requests.get("http://example.com/api1")
    response_api2 = requests.get("http://example.com/api2")
    
    # Prepare a combined result in JSON format
    combined_result = json.dumps({
        "api1": response_api1.json(),
        "api2": response_api2.json()
    }, indent=2)
    
    # Print the combined JSON result
    print(combined_result)

    # Disable httpretty after the requests are made
    httpretty.disable()
    httpretty.reset()

def main():
    setup_mock_api_endpoints()
    fetch_data_from_apis()

if __name__ == "__main__":
    main()
