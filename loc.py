from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pyttsx3
import webbrowser
import os

app = Flask(__name__)
CORS(app)

# OpenRouteService API Key
ORS_API_KEY = "5b3ce3597851110001cf62489c92d2f752544257b50b50e1a09f2288"

# Destination (Bangalore)
DEST_LAT = 13.0827
DEST_LON = 80.2707

# Store last received location
last_location = {"latitude": None, "longitude": None}

@app.route('/receive_location', methods=['POST'])
def receive_location():
    global last_location

    data = request.json

    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if latitude is None or longitude is None:
        return jsonify({"error": "Invalid coordinates"}), 400

    print(f"Received Coordinates: Latitude {latitude}, Longitude {longitude}")

    # Update last known location
    last_location["latitude"] = latitude
    last_location["longitude"] = longitude

    # Text-to-speech for location received
    engine = pyttsx3.init()
    engine.say("Location received. Fetching directions.")
    engine.runAndWait()

    # Fetch navigation instructions
    route_url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "coordinates": [[longitude, latitude], [DEST_LON, DEST_LAT]],
        "instructions": True
    }

    response = requests.post(route_url, json=payload, headers=headers)

    if response.status_code == 200:
        navigation_data = response.json()
        
        # Extract step-by-step navigation
        steps = navigation_data['features'][0]['properties']['segments'][0]['steps']
        directions = []
        for step in steps:
            direction = {
                "distance": step["distance"],  # Distance in meters
                "direction": step["instruction"],  # Navigation instruction
                "time": step["duration"]  # Duration in seconds
            }
            directions.append(direction)

        print(directions)

        # Speak the first few steps
        try:
            steps_to_speak = " . ".join([step["direction"] for step in directions[:3]])
            engine.say(steps_to_speak)
            engine.runAndWait()
        except RuntimeError as e:
            print(f"Text-to-speech error: {e}")

        return jsonify({"message": "Updated navigation", "current_location": last_location, "directions": directions})

    else:
        return jsonify({"error": "Failed to call navigation API", "details": response.text}), 500

if __name__ == '__main__':
    # Open the HTML file in default browser
    webbrowser.open("file://" + os.path.realpath("index.html"))
    app.run(host="0.0.0.0", port=5002, debug=True)
    