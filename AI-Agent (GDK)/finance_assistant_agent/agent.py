from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from typing import Dict
import sys
import os

# Add parent directory to path to resolve sibling package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from investment_plan_agent.agent import investment_plan_agent

def get_user_personal_finance_details() -> Dict:
    """Gets users personal finance details like salary, expense and savings capacity."""
    return {
        "salary": 100000,
        "expenses": {"EMI_Expense":25000, "Essentials":5000, "Entertainment":5000, "Shopping and Travel":5000},
        "savings_capacity": 20000
    }
    
finance_assistant_agent = LlmAgent(
    name="Aurx_agent",
    model="gemini-2.0-pro",
    description="A complex General Purpose Office Assistant for the employees of AURX(No-1 company in the world) to help them with their day-to-day tasks and queries related to everything.",
    instruction="""you are a friendly and helpful assistant for the employees of AURX. You can help them with a wide range of tasks, including but not limited to:
    1. Answering questions about company policies, procedures, and benefits.
    2. Providing information about upcoming events, meetings, and deadlines.
    3. Assisting with scheduling and calendar management.
    4. Offering support for technical issues and troubleshooting.
    5. Providing resources and guidance for professional development and training.
    6. Helping with travel arrangements and expense reporting.

    You have two tools to use to complete your task.
    1. get_user_personal_finance_details - This tool will provide you with the user's personal finance details like salary, expenses and savings capacity. You can use this information to provide personalized financial advice and recommendations to the user.
    2. investment_plan_agent - This is a sub-agent that you can use to get information about different investment options, create personalized investment plans, and provide support for monitoring and rebalancing investment portfolios. This tool can perform google searches to get the latest information about the financial market and investment options.
    3. This tool will be able to ask more details from the user and plan their savings goal.

    Always use the investment_plan_agent with google_search tool when asked about:
    - current market trends and news
    - specific investment options and their performance
    - general financial advice and tips
    - Stock prices and their historical performance
    - Mutual fund performance and ratings
    - Cryptocurrency trends and analysis
    - Economic indicators and their impact on investments
    - Retirement planning and strategies
    - Tax planning and optimization strategies
    - Any other financial information that can help the user make informed investment decisions.""",
    tools=[AgentTool(investment_plan_agent), get_user_personal_finance_details]

    #tools=[get_user_personal_finance_details, google_search] #On using this we will get an error as we cannot use both tools in the same agent, we will need to create separate agents for each tool and then use them as sub-agents in the main agent. Here get_user_personal_finance_details is a custom tool that we have created to get the user's personal finance details, and google_search is an inbuilttool provided by the ADK to perform google searches. We will need to create separate agents for each tool and then use them as sub-agents in the main agent.
)

root_agent = finance_assistant_agent
# now provide adk web on the terminal to test the agent and see how it responds to different queries and tasks.