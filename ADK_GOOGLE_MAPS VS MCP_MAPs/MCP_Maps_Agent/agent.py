import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
from google.genai import types

load_dotenv()

MAPS_API_KEY = os.getenv("MAPS_API_KEY")

MAPS_MCP_URL = "https://mapstools.googleapis.com/mcp"

maps_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url=MAPS_MCP_URL,
        headers={"X-Goog-Api-Key": MAPS_API_KEY}
    )
)

root_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="maps_mcp_agent",
    instruction=(
        "You are a helpful Maps assistant. \n"
        "Use the MCP-provided Maps tools to find places and get directions.\n"
        "When directions are requested, include a Google Maps link in the final answer."
    ),
    tools=[maps_toolset]
)
