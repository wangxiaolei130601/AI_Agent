import os
os.environ["DASHSCOPE_API_KEY"] = "sk-ecdbc4a7a2b140c69b04c6d5953a11d6"


# 1. 定义1个工具：执行python程序
from langchain_experimental.tools import PythonREPLTool
# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()
# 1. 定义1个工具：执行数据库查询
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.llms import Tongyi
db = SQLDatabase.from_uri("mysql+pymysql://root:123456@127.0.0.1/testdb")
mysql_toolkit = SQLDatabaseToolkit(db=db, llm=Tongyi(temperature=0))


# 2. 定义创建agent函数
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
def create_agent(
    llm: Tongyi, tools: list, system_prompt: str
):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


# 3. 把agent返回结果转化为自然语言
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


# 4. 创建Supervisor(路由) Agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema.output_parser import StrOutputParser
import json
members = ["MySQLer", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm = Tongyi(model="qwen-plus", temperature=0.0)

# 方法1：通义千问没有bind_functions函数
# supervisor_chain = (
#     prompt
#     | llm.bind_functions(functions=[function_def], function_call="route")
#     | JsonOutputFunctionsParser()
# )
# 方法2
output_parser = StrOutputParser()
supervisor_chain = (
    prompt
    | llm.bind(functions=[function_def], function_call="route")
    | output_parser
    | json.loads
)
# supervisor_chain = llm.with_fallbacks([initial_chain])

# 方法3
# functionCallingModel = llm.bind(functions=[function_def], function_call="route")
# supervisor_chain = prompt.pipe(functionCallingModel).pipe(JsonOutputFunctionsParser())

# 5. 构建graph：定义state和agent节点
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools
from langgraph.graph import StateGraph, END
# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


from langchain.prompts.prompt import PromptTemplate
# 设置sql agent提示词
mysql_template = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

The SQL query should be outputted plainly, do not surround it in quotes or anything else.

Only use the tables listed below.

{table_info}

Question: {input}"""

PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=mysql_template
)

# mysql_agent = create_agent(llm, [mysql_tool], mysql_template)
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
mysql_agent = create_sql_agent(llm, toolkit=mysql_toolkit, agent_type=AgentType.OPENAI_FUNCTIONS, verbose=True)
# from langchain_experimental.sql import SQLDatabaseChain
# mysql_agent = SQLDatabaseChain(llm=llm, database=db, verbose=True, prompt=PROMPT)
mysql_node = functools.partial(agent_node, agent=mysql_agent, name="MySQLer")

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
code_agent = create_agent(llm, [python_repl_tool], "You may generate safe python code to analyze data and generate charts using matplotlib.")
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

workflow = StateGraph(AgentState)
workflow.add_node("MySQLer", mysql_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_chain)


# 6. 连接图中的边
for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.set_entry_point("supervisor")

graph = workflow.compile()


# 7. 运行图
for s in graph.stream(
    {
        "messages": [
            # HumanMessage(content="Code hello world and print it to the terminal")
            HumanMessage(content="How many students are there?")
        ]
    }
):
    if "__end__" not in s:
        print(s)
        print("----")