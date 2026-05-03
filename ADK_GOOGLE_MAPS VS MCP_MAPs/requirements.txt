import os
import requests
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

load_dotenv()
MAPS_API_KEY = os.getenv("MAPS_API_KEY")

#Google Maps- Places Nearby Search API
def search_places_on_route(origin: str, destination: str, place_type: str) -> dict:
    """Search for places along the route between two locations.
    
    Args:
    origin: Starting location
    destination: End location
    place_type: What to search for (e.g., 'gas station), 'restaurant", 'hotel', 'atm')"""
    
    #Get route to find midpoint
    dir_response = requests.get("https://maps.googleapis.com/maps/api/directions/json", params={
        "origin": origin,
        "destination": destination,
        "key": MAPS_API_KEY},
                    timeout=20)
    dir_data = dir_response.json()
    
    if dir_data.get("status") != "OK":
        return {"error": "could not find route"}
    
    #Get midpoint from route
    steps = dir_data["routes"][0]["legs"][0]["steps"]
    mid = steps[len(steps) // 2]["end_location"]
    
    #Search for places near midpoint
    places_response = requests.get("https://maps.googleapis.com/maps/api/place/nearbysearch/json", params={
        "location": f"{mid['lat']},{mid['lng']}",
        "radius": 5000,
        "keyword": place_type,
        "key": MAPS_API_KEY},
                    timeout=20)

#Google maps- Directions API calls
def get_directions(origin: str, destination: str, mode: str= "driving") -> dict:
    """Get route summary using Google Directions API."""
    url = "https://maps.googleapis.com/maps/api/directions/json"
    r= requests.get(url, params={
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "key": MAPS_API_KEY},
                    timeout=20)
    r.raise_for_status()
    data = r.json()
    
    if data.get("status") != "OK" or not data.get("routes"):
        return {"status": data.get("status"), "error": data.get("error_message", "No routes found")}
    
    leg = data["routes"][0]["legs"][0]
    return{
        "origin": leg["start_address"],
        "destination": leg["end_address"],
        "distance": leg["distance"]["text"],
        "duration": leg["duration"]["text"],
        "google_maps_link": f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}&distance={leg['distance']['text']}&duration={leg['duration']['text']}"
    }        
#Agent
directions_tool = FunctionTool(get_directions)
places_tool = FunctionTool(search_places_on_route)

root_agent = LlmAgent(model="gemini-2.0-flash", name="Maps_Agent",
                      instruction=("You are a helpful Maps assistant.\n"
                                   "Use appropriate tools to answer user queries about directions and routes.\n"
                                   "1) TOOL: Get directions between two locations\n"
                                   "2) TOOL: Search for places along the route between two locations\n"
                                   "Always return a short answer + the Google Maps link if applicable"),
                      tools=[directions_tool, places_tool])
                      
                      
#Here if you look at the instruction, the agent is being told to use the tool to get directions between two locations and return a short answer along with the Google Maps link if applicable. The agent will use the get_directions function to fetch the required information and provide a response based on the user's query.
#We have mentioned all the details and fortunes of the API- example: How to hit an API? What if it fails? What to do on Error Handling? What could be the possible response?- we wrote all these like an API engineer. 
#When we try to build similar ot other agents similar setup should be followed rather than reusing it. So there is no reusability
# #To solve this issue we use MCP