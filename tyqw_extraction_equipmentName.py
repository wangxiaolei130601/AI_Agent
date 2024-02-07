# 引入环境变量：通义千问大模型API_KEY
import os
os.environ["DASHSCOPE_API_KEY"] = "sk-ecdbc4a7a2b140c69b04c6d5953a11d6"


from typing import Optional, Sequence
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.llms import Tongyi
from langchain.output_parsers.enum import EnumOutputParser
from enum import Enum


class EquipmentName(Enum):
    """Define the name of equipment you have to identify in a text."""

    Primary_Crusher_Dust_Collector = "粗碎除尘器"
    Medium_Crusher_Dust_Collector = "中碎筛分除尘器"
    Fine_Crusher_Dust_Collector = "细碎除尘器"


class Failure(BaseModel):
    """Identifying information about the failure of some equipment in a text."""

    equipment_name: EquipmentName


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Failure)

# Prompt
prompt = PromptTemplate(
    template="Which equipment does the user mention?\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Run
model = Tongyi(model="qwen-max-1201", temperature=0)
while True:
    query = input("用户问题: ")
    format_prompt = prompt.format_prompt(query=query)
    output = model(format_prompt.to_string())
    format_output = parser.parse(output)
    print("设备名称: " + format_output.equipment_name.value)
    print("-----------------------------------\n")

