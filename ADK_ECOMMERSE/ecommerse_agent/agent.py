import sys
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from catalog_agent.agent import catalog_agent
from checkout_agent.agent import checkout_agent
from tracking_agent.agent import tracking_agent
from config import AGENT_MODEL
def save_user_info(tool_context: ToolContext,name: str,email: str,mobile:str):
    
    tool_context.state["name"] = name
    tool_context.state["email"] = email
    tool_context.state["mobile"] = mobile
    
root_agent = LlmAgent(
    name="ecommerce_agent",
    description="An ecommerce agent that manages the ecommerce workflow",
    model=AGENT_MODEL,
    instruction="""
    Role: You are an ecommerce agent that manages the ecommerce workflow. You will be responsible for handling customer inquiries, processing orders, and providing support.
    Workflow:
        -Greet the user and give a brief introduction about yourself on how can you help and then start gathering the user details as mentioned below. Do not directly start gathering user information.
        -If you do not know, ask for the user's name, email and mobile number. Ask only one information at a time.
        -Once you have the above information, call the save_user_info() tool to save these information.
        -Then understand the user's intent. Are they looking for new purchase or track an existing order.
        -Based on the user's request and route it to ONE of your sub-agents:
            -catalog_agent - For New purchases, questions about products, prices, availability etc.
            -checkout_agent - For checkout of items in cart.
            -tracking_agent - For tracking existing orders.
    Rules:
        -NEVER answer the question yourself. Always delegate to exactly one sub-agent.
        -If the user's message clearly matches one category, immediately call that agent.
        -If you are unsure, ask a short clarifying question instead of guessing.
        -After a sub-agent responds, you may send that response back as-is to the user, without adding extra content.        """,
        tools=[save_user_info],
        sub_agents=[catalog_agent, checkout_agent, tracking_agent]
)
