import json

from langchain.chains import LLMChain
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.llms import Tongyi
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

os.environ["DASHSCOPE_API_KEY"] = "sk-ecdbc4a7a2b140c69b04c6d5953a11d6"


class Student(BaseModel):
    "Give the name and the age of some student."
    name: str = Field(description="学生的姓名")
    age: int = Field(description="学生的年龄")


student_query = "告诉我一个学生的信息"

parser = PydanticOutputParser(pydantic_object=Student)
output_format1 = parser.get_format_instructions()
prompt1 = PromptTemplate(
    template="回答下面问题.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": output_format1 + "用中文回答"},
)

# function_Student = {
#     "description": "Give the name and the age of some student.",
#     "properties": {
#         "name": {
#             "description": "学生的姓名",
#             "title": "Name",
#             "type": "string"
#         },
#         "age": {
#             "description": "学生的年龄",
#             "title": "Age",
#             "type": "integer"
#         }
#     },
#     "required": ["name", "age"]
# }
function_Student = {
    "description": "Give the name and the age of some student."
    + " Format the arguments as a JSON object.",
    "parameters": [
        {
            "name": "name",
            "description": "the name of student.",
            "required": True,
            "type": "string",
        },
        {
            "name": "age",
            "description": "the age of student.",
            "required": True,
            "type": "int",
        },
    ],
}
output_format2 = (
     "The output should be formatted as a JSON instance that conforms to the JSON schema below."
     "Here is the output schema:\n"
                 ) + json.dumps(function_Student)
prompt2 = PromptTemplate(
    template="回答下面问题.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": output_format2},
)

llm = Tongyi(temperature=0.0)
# _input1 = prompt1.format_prompt(query=student_query)
# output1 = llm(_input1.to_string())
# print(output1)
# print(parser.parse(output1))

chain = LLMChain(prompt=prompt2, llm=llm)
output2 = chain.invoke({"query": "给我一个学生信息"})
print(output2)
