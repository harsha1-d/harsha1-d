import sys
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from order_summary_agent.agent import order_summary_agent
from config import AGENT_MODEL

def save_shipping_address(tool_context: ToolContext, address: str):
    tool_context.state["shipping_address"] = address

checkout_agent = LlmAgent(
    name="checkout_agent",
    description="A checkout agent that helps users complete purchases.",
    model=AGENT_MODEL,
    instruction="""
    Your scope:
        - Help users review cart items and complete checkout.
        - Ask for any missing delivery or payment details one at a time.
        - Keep responses short and friendly.
    Rules:
        - Stay only in checkout/order placement.
        - Do not answer catalog browsing or tracking questions.
    """,
    tools=[save_shipping_address],
    sub_agents=[order_summary_agent]
)
