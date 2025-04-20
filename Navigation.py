from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import webbrowser
import os

app = Flask(__name__)
CORS(app)

# OpenRouteService API Key
ORS_API_KEY = "5b3ce3597851110001cf62489c92d2f752544257b50b50e1a09f2288"


# Store last received location
last_location = {"latitude": None, "longitude": None}
last_summary = None
destination_location = {"latitude": None, "longitude": None}

@app.route('/receive_location', methods=['POST'])
def receive_location():
    global last_location, last_summary, destination_location

    data = request.json

    latitude = data.get("latitude")
    longitude = data.get("longitude")
    dest_lat = data.get("dest_latitude")
    dest_lon = data.get("dest_longitude")

    if latitude is not None and longitude is not None:
        print(f"Received Current Coordinates: Latitude {latitude}, Longitude {longitude}")
        last_location["latitude"] = latitude
        last_location["longitude"] = longitude
    if dest_lat is not None and dest_lon is not None:
        print(f"Received Destination Coordinates: Latitude {dest_lat}, Longitude {dest_lon}")
        destination_location["latitude"] = dest_lat
        destination_location["longitude"] = dest_lon

    if None in (last_location["latitude"], last_location["longitude"],
                destination_location["latitude"], destination_location["longitude"]):
        return jsonify({"message": "Waiting for complete coordinates"}), 200

    route_url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "coordinates": [
            [last_location["longitude"], last_location["latitude"]],
            [destination_location["longitude"], destination_location["latitude"]]
        ],
        "instructions": True
    }

    response = requests.post(route_url, json=payload, headers=headers)

    if response.status_code == 200:
        navigation_data = response.json()
        steps = navigation_data['features'][0]['properties']['segments'][0]['steps']
        directions = [{
            "distance": step["distance"],
            "direction": step["instruction"],
            "time": step["duration"]
        } for step in steps]

        last_summary = " . ".join([step["direction"] for step in directions[:3]])

        return jsonify({
            "message": "Updated navigation",
            "current_location": last_location,
            "directions": directions,
            "summary": last_summary
        })
    else:
        return jsonify({"error": "Failed to call navigation API", "details": response.text}), 500

@app.route('/last_summary', methods=['GET'])
def get_summary():
    return jsonify({"summary": last_summary})

if __name__ == '__main__':
    # Open the HTML file in default browser
    # webbrowser.open("file://" + os.path.realpath("index.html"))
    app.run(host="0.0.0.0", port=5002, debug=True)