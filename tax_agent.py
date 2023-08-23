from dotenv import load_dotenv
#from langchain.utilities import SerpAPIWrapper
from langchain.agents import load_tools
from langchain.agents import  initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from load_data_to_database import load_db_from_disk
from langchain.tools import BaseTool

#including envirnoment variables
load_dotenv()

# create llm object
llm= OpenAI(temperature=0)

#create tools
# create retrivial tool of individul taxes vector db
db =load_db_from_disk()
individul_tax= RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=db.as_retriever())
tax= Tool(
    name='Individual Tax QA',
    func= individul_tax.run,
    description= "useful when answering south african income tax questions for an individual. Input should be fully formed sentence"

)
tool_names_list =['serpapi','llm-math',]
tools = load_tools(tool_names=tool_names_list,llm=llm)
tools.append(tax)


# initialize agent with tools, llm, type of agent
agent = initialize_agent(tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


if __name__ == "__main__":
    query="Who is Leo DiCaprio's girlfriend? What is her current age raised to the 2 power?"
    query_2= "would Leo DiCaprio be required to pay taxes in south africa"
    response=agent.run(query)
    print(f'Question 1:{query}\n ',response)
    response=agent.run(query_2)
    print(f'Question 2:{query_2}\n ',response)
