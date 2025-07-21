from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai

import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key= os.getenv("OPENAI_API_KEY")

## Web Search Agent
web_search_agent = Agent(
    name ="Web search Agent",
    role ="Search the web for the infromation",
    model = Groq(model_name = "llama3-groq-70b-8192-tool-use-preview"),
    tools =[DuckDuckGo()],
    instructions= ["Always include the source"],
    show_tools_calls=True,
    markdown=True,
)

##Financial Agent
finance_agent = Agent(
    name="Financial AI Agent",
    model=Groq(model_name = "llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True),],
    instructions=["Use tables to display the data."],
    show_tools_calls=True,
    markdown=True,
)

multi_ai_agent =Agent(
    team= [web_search_agent, finance_agent],
    instructions= ["Always include sources", "Use tables to display the data."],
    show_tools_calls= True,
    markdown=True,
)

multi_ai_agent.print_response("Summarise analyst recommendation and share the latest news fr NVDIA", stream = True)