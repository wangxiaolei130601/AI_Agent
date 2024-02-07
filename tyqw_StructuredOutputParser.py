from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema.output_parser import StrOutputParser
from langchain.llms import Tongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
import json
import os
os.environ["DASHSCOPE_API_KEY"] = "sk-ecdbc4a7a2b140c69b04c6d5953a11d6"


response_schemas = [
    ResponseSchema(name="name", description="学生的姓名"),
    ResponseSchema(name="age", description="学生的年龄")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="回答下面问题.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)

model=Tongyi(temperature=1.0)
_input = prompt.format_prompt(question="给我一个女孩的名字?")
output = model(_input.to_string())
print(output)
print(output_parser.parse(output))