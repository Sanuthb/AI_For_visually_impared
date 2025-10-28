from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pyttsx3
import os

app = Flask(__name__)
CORS(app)

# OpenRouteService API Key
# NOTE: Using a placeholder key from your previous files. Ensure this key is valid.
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjFlZDhmZjdmNTc3MTQ4Y2E4MTZkYmRkMTM3MGQ5OTQ5IiwiaCI6Im11cm11cjY0In0="

# Store last received location
last_location = {"latitude": None, "longitude": None}
destination_location = {"latitude": None, "longitude": None}
full_route_steps = []
current_step_index = -1 # -1 means no route active

def announce(text):
    """Helper function to speak text using pyttsx3."""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Text-to-speech error: {e}")

@app.route('/receive_location', methods=['POST'])
def receive_location():
    global last_location, destination_location, full_route_steps, current_step_index

    data = request.json
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    dest_lat = data.get("dest_latitude")
    dest_lon = data.get("dest_longitude")
    is_new_destination = False

    # 1. Update Current Location
    if latitude is not None and longitude is not None:
        last_location.update({"latitude": latitude, "longitude": longitude})
    
    # 2. Check for New Destination
    if dest_lat is not None and dest_lon is not None:
        if (destination_location["latitude"] != dest_lat or
                destination_location["longitude"] != dest_lon):
            destination_location.update({"latitude": dest_lat, "longitude": dest_lon})
            is_new_destination = True
            full_route_steps = [] # Reset route
            current_step_index = -1
            print("New Destination Set. Clearing route.")

    if None in (last_location["latitude"], last_location["longitude"],
                destination_location["latitude"], destination_location["longitude"]):
        return jsonify({"message": "Waiting for complete coordinates"}), 200

    # 3. Calculate Route (ONLY if it's a new destination or route is missing)
    if is_new_destination or not full_route_steps:
        announce("Coordinates received. Calculating new navigation route.")
        
        route_url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
        headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
        payload = {
            "coordinates": [[last_location["longitude"], last_location["latitude"]],
                            [destination_location["longitude"], destination_location["latitude"]]],
            "instructions": True
        }

        response = requests.post(route_url, json=payload, headers=headers)

        if response.status_code == 200:
            navigation_data = response.json()
            try:
                # Get the full geometry (coordinates) for accurate turn location
                geometry = navigation_data['features'][0]['geometry']['coordinates']
                steps = navigation_data['features'][0]['properties']['segments'][0]['steps']
                
                # Store the full route steps, including the next turn coordinates
                full_route_steps = []
                for step in steps:
                    end_coord_index = step['way_points'][1]
                    # Coordinates are [longitude, latitude] in ORS geometry
                    turn_lon, turn_lat = geometry[end_coord_index] 
                    
                    full_route_steps.append({
                        "distance": step["distance"],
                        "direction": step["instruction"],
                        "time": step["duration"],
                        "turn_latitude": turn_lat,        # <--- Used by client for proximity check
                        "turn_longitude": turn_lon        # <--- Used by client for proximity check
                    })
                
                current_step_index = 0
                first_step = full_route_steps[0]
                announce(f"Route calculated. The first step is: {first_step['direction']}")
            
            except (KeyError, IndexError) as e:
                announce("Failed to parse route data. Route calculation failed.")
                return jsonify({"error": f"Failed to parse route data. Details: {e}"}), 500
        else:
            announce("Navigation API failed to calculate route.")
            return jsonify({"error": f"Failed to call navigation API. Status: {response.status_code}"}), 500

    # 4. Respond with Current Step (Continuous Monitoring Data)
    current_step = None
    if 0 <= current_step_index < len(full_route_steps):
        current_step = full_route_steps[current_step_index]
        
    # If the user is at the destination, return a finished state
    elif current_step_index >= len(full_route_steps) and len(full_route_steps) > 0:
        return jsonify({"message": "Destination reached.", "current_step_index": current_step_index, "current_step": None}), 200

    return jsonify({
        "message": "Route active for monitoring.",
        "current_location": last_location,
        "current_step": current_step, 
        "current_step_index": current_step_index,
        "total_steps": len(full_route_steps)
    })

@app.route('/next_step', methods=['POST'])
def next_step():
    """
    Endpoint called by the client (mobile app) when it detects the user has completed a turn
    (i.e., proximity to the next turn coordinate is met).
    """
    global current_step_index, full_route_steps
    
    if current_step_index < len(full_route_steps) - 1:
        current_step_index += 1
        new_step = full_route_steps[current_step_index]
        distance_in_meters = round(new_step['distance'])
        
        instruction_to_speak = f"Advanced. Travel {distance_in_meters} meters. Then, {new_step['direction']}"
        announce(instruction_to_speak)
        
        return jsonify({
            "message": "Advanced to next step", 
            "current_step": new_step,
            "current_step_index": current_step_index
        })
    elif current_step_index == len(full_route_steps) - 1 and len(full_route_steps) > 0:
        announce("You have reached the final step. The destination is near.")
        # Advance the index one last time to signal the end state
        current_step_index += 1 
        return jsonify({"message": "At final step (destination reached)", "current_step_index": current_step_index})
    else:
        return jsonify({"message": "No active route. Please set a destination via /receive_location."}), 200

# NOTE: The /last_summary endpoint was removed as it relied on an undefined variable.

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True)