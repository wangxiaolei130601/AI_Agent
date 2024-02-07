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


class Person(BaseModel):
    person_name: str
    person_height: int
    person_hair_color: str
    dog_breed: Optional[str]
    dog_name: Optional[str]


class People(BaseModel):
    """Identifying information about all people in a text."""

    people: Sequence[Person]


# Run
query = """Alex is 5 feet tall. Claudia is 1 feet taller Alex and jumps higher than him. Claudia is a brunette and Alex is blonde."""

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=People)

# Prompt
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Run
_input = prompt.format_prompt(query=query)
model = Tongyi(temperature=0)
output = model(_input.to_string())
format_output = parser.parse(output)
print(format_output)
