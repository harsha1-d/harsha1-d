from google.adk.agents import LlmAgent

from config import AGENT_MODEL


order_summary_agent = LlmAgent(
    name="order_summary_agent",
    description="An order summary agent that summarizes cart and shipping details.",
    model=AGENT_MODEL,
    instruction="""
    Your scope:
        - Summarize the user's cart, quantity, price, and shipping address from state.
        - Ask for any missing cart or shipping detail one at a time.
        - Keep the summary short and clear.
        - You should mention:
            {name} {email} {mobile}
            {category} {item} {quantity} {price} {shipping_address}
    Rules:
        - Do not browse products or track orders.
        - Do not claim payment is complete unless the user has clearly confirmed it.
    """,
)
