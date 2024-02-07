import os
os.environ["DASHSCOPE_API_KEY"] = "sk-ecdbc4a7a2b140c69b04c6d5953a11d6"

from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
import json


# 1. 初始化大模型
llm = Tongyi(temperature=0.0)


tools_name = ["MySQLer, Caller"]
system_prompt = (
    "Given a user request, respond with one of the following tools. Each tool will perform a task."
    # "Here are the tools and their descriptions you can choose:\n{tools_desc}"
    "Here are the tools and their capabilities:"
    "1. MySQLer: Search database then return the answer."
    "2. Caller: Call function then return the result."
)
output_def = {
    "name_for_model": "route",
    "description_for_model": "Select the tool responding with user request."
                             + " Format the arguments as a JSON object.",
    "parameters": [
        {
            "name": "next",
            "description": "the next tool to respond with user request.",
            "required": True,
            "type": "string",
            "enum": tools_name
        }
    ]
}

output_format = ("The output should be formatted as a JSON instance that conforms to the JSON schema below."
                 "Here is the output schema:\n") + json.dumps(output_def) + "用中文回答"
prompt = ChatPromptTemplate.from_messages(
    messages=[
        ("system", system_prompt),
        ("human", "{input}"),
        (
            "system",
            "Given the user request above, which tool should work next?"
            " Only select one of: {tools_name}",
        ),
    ],
).partial(tools_name=str(tools_name), format_instructions=output_format)

supervisor = LLMChain(prompt=prompt, llm=llm, output_key="next")

respond = supervisor.invoke({"input": "Search database and answer how many students there are"})
print(respond)
print("---")

respond = supervisor.invoke({"input": "Call functions."})
print(respond)
print("---")
