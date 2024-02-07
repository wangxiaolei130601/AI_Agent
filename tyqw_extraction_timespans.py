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


class TimePeriod(BaseModel):
    """Identifying the start day and the end day mentioned in a text."""

    start_day: str = Field(description="Start day of a period. The format should be in the format YYYY-MM-DD.")
    end_day: str = Field(description="End day of a period. The format should be in the format YYYY-MM-DD.")


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=TimePeriod)

# Run
model = Tongyi(model="qwen-plus", temperature=0)
while True:
    # Identify and output start day and end day of the time frame mentioned by user. Time is based on solar calendar.
    prompt_template = input("Prompt template: ")
    # Prompt
    prompt = PromptTemplate(
        template=prompt_template + "Today is 2024-02-07.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    while True:
        query = input("User query: ")
        if query == "break":
            break
        format_prompt = prompt.format_prompt(query=query)
        output = model(format_prompt.to_string())
        format_output = parser.parse(output)
        print("Start day: " + format_output.start_day)
        print("End day: " + format_output.end_day)
        print("-----------------------------------\n")

