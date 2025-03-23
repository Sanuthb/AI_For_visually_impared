from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Set up Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")  # Avoid detection as bot
driver = webdriver.Chrome(options=options)

def get_current_location():
    """Fetches the current location using Google Maps."""
    
    # Open Google Maps
    driver.get("https://www.google.com/maps")
    time.sleep(3)  # Wait for page to load

    # Click on "My Location" button
    try:
        my_location_button = driver.find_element(By.XPATH, "//button[contains(@aria-label, 'Your location')]")
        my_location_button.click()
        time.sleep(5)  # Wait for the map to update
    except Exception as e:
        print("Error: Unable to fetch location. Make sure location access is allowed.")
        driver.quit()
        return None

    # Extract coordinates from URL
    current_url = driver.current_url
    try:
        parts = current_url.split("/@")[1].split(",")  # Extract lat, lon
        latitude, longitude = parts[0], parts[1]
        print(f"Current Location: Latitude: {latitude}, Longitude: {longitude}")
        return latitude, longitude
    except Exception as e:
        print("Error extracting coordinates:", e)
        return None

# Get current location
get_current_location()

# Close browser
driver.quit()
