import requests
from datetime import datetime, timedelta

# Constants
API_KEY_WEATHER = 'your_openweathermap_api_key'
API_KEY_PLACES = 'your_googleplaces_api_key'
LOCATION = 'Gdańsk, PL'
WEATHER_ENDPOINT = 'https://api.openweathermap.org/data/2.5/onecall'
PLACES_ENDPOINT = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'

def get_weather_data(lat, lon):
    params = {
        'lat': lat,
        'lon': lon,
        'appid': API_KEY_WEATHER,
        'exclude': 'current,minutely,hourly,alerts',
        'units': 'metric'
    }
    response = requests.get(WEATHER_ENDPOINT, params=params)
    return response.json()

def get_places_data(lat, lon):
    params = {
        'location': f"{lat},{lon}",
        'radius': '5000',  # Search within 5 kilometers
        'key': API_KEY_PLACES
    }
    response = requests.get(PLACES_ENDPOINT, params=params)
    return response.json()

def main():
    # Gdańsk coordinates
    lat, lon = 54.352025, 18.646638
    
    # Fetch weather data
    weather_data = get_weather_data(lat, lon)
    
    # Fetch places data
    places_data = get_places_data(lat, lon)
    
    today = datetime.now()
    end_of_year = datetime(today.year, 12, 31)
    delta = timedelta(days=1)
    
    while today <= end_of_year:
        if today.weekday() == 5:  # Saturday
            weekend_weather = weather_data['daily'][0]  # Simplistic approach, adjust indexing as needed
            print(f"Weekend of {today.strftime('%Y-%m-%d')}:")
            print(f"- Weather: {weekend_weather['weather'][0]['description']}. Temp: {weekend_weather['temp']['day']}°C")
            sites = places_data['results'][:5]  # Example: take top 5 places
            for site in sites:
                print(f"- Visit: {site['name']}")
            today += timedelta(days=1)  # Move to Sunday
        today += delta

if __name__ == '__main__':
    main()
