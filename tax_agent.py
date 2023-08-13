from dotenv import load_dotenv
#from langchain.utilities import SerpAPIWrapper
from langchain.agents import load_tools
from langchain.agents import  initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI



#including envirnoment variables
load_dotenv()

# create llm object
llm= OpenAI(temperature=0)

#create tools
tool_names_list =['serpapi','llm-math']
tools = load_tools(tool_names=tool_names_list,llm=llm)

# initialize agent with tools, llm, type of agent

agent = initialize_agent(tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


if __name__ == "__main__":
    query="Who is Leo DiCaprio's girlfriend? What is her current age raised to the 2 power?"
    response=agent.run(query)
    print(response)
