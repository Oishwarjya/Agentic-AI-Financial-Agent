from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
from dotenv import load_dotenv
import openai

import os
openai.api_key = os.getenv("OPENAI_API_KEY")

from phi.playground import Playground, serve_playground_app

##Load environment variables
load_dotenv()

phi.api= os.getenv("PHI_API_KEY")

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

app= Playground(agents =[finance_agent, web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app", reload=True)
