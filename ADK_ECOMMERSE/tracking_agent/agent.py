from google.adk.agents import LlmAgent
from config import AGENT_MODEL

tracking_agent = LlmAgent(
    name="tracking_agent",
    description="A tracking agent that helps users track existing orders.",
    model=AGENT_MODEL,
    instruction="""
    Your scope:
        - Help users track an existing order.
        - Ask for the order ID if it is missing.
        - Keep responses short and friendly.
    Rules:
        - Stay only in order tracking.
        - Do not answer catalog browsing or checkout questions.
    """,
)
