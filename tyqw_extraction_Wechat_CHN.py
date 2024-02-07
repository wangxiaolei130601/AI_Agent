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
import datetime


class EquipmentName(Enum):
    # """Define the name of equipment you have to identify in a text."""

    # Primary_Crusher_Dust_Collector = "primary crusher dust collector"
    # Medium_Crusher_Dust_Collector = "medium crusher dust collector" or "sifting dust collector" or ("medium crusher "
    #                                                                                                 "and sifting dust"
    #                                                                                                 " collector")
    # Fine_Crusher_Dust_Collector = "fine crusher dust collector"
    Primary_Crusher_Dust_Collector = "粗碎除尘器"
    Medium_Crusher_Dust_Collector = "中碎除尘器" or "筛分除尘器" or "中碎筛分除尘器"
    Fine_Crusher_Dust_Collector = "细碎除尘器"


class FailureType(Enum):
    # """Define the type of equipment failure you have to identify in a text."""

    # Loss_Pres_Failure = "Differential pressure failure"
    # Delivery_Failure = "Ash delivery failure"
    Loss_Pres_Failure = "差压过高"
    Delivery_Failure = "输灰不畅"


class TimePeriod(BaseModel):
    # """Identifying the start date and the end date mentioned in a text."""

    # start_day: str = Field(description="Start time of a period. The format should be in the format YYYY-MM-DD.")
    # end_day: str = Field(description="End time of a period. The format should be in the format YYYY-MM-DD.")
    start_day: str = Field(description="一个时间范围的开始时间，时间格式使用YYYY-MM-DD。")
    end_day: str = Field(description="一个时间范围的结束时间，时间格式使用YYYY-MM-DD。")


class Failure(BaseModel):
    # """Identifying information about the failure of some equipment in a text."""

    # equipment_name: Optional[EquipmentName] = Field(
    #     description="Name of equipment with failure", default=None
    # )
    # failure_type: Optional[FailureType] = Field(
    #     description="Type of equipment failure", default=None
    # )
    # time_period: Optional[TimePeriod] = Field(
    #     description="A time period during which equipment failure occurs", default=None
    # )
    equipment_name: Optional[EquipmentName] = Field(
        description="故障设备的名称", default=None
    )
    failure_type: Optional[FailureType] = Field(
        description="故障类型", default=None
    )
    time_period: Optional[TimePeriod] = Field(
        description="发生故障的时间范围，使用阳历。", default=None
    )


# class Failures(BaseModel):
#     """Identifying information about all failures of equipment in a text."""
#
#     failures: Sequence[Failure]


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Failure)

today = datetime.datetime.today()
today_string = "%d-%d-%d" % (today.year, today.month, today.day)
# Prompt
prompt = PromptTemplate(
    # template="Identifying name of equipment with failure, type of equipment failure and time period during which"
    #          " failure occurs in the query. Today is %s.\n{format_instructions}\n{query}\n"
    #          % today_string,
    template="从用户输入中识别故障设备的名称、故障类型和发生故障的时间范围。今天的日期是%s.\n{format_instructions}\n{query}\n"
             % today_string,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions() + "用中文回答"},
)

# Run
model = Tongyi(model="qwen-max-1201", temperature=0)
# for i in range(len(query_list)):
#     query = query_list[i]
#     format_prompt = prompt.format_prompt(query=query)
#     output = model(format_prompt.to_string())
#     format_output = parser.parse(output)
#     print("%d.用户问题: %s" % (i+1, query))
#     print("提取故障设备、故障类型如下：")
#     print("1)故障设备: " + format_output.equipment_name)
#     print("2)故障类型: " + format_output.failure_type.value)
#     print("-----------------------------------\n")
while True:
    query = input("用户问题: ")
    format_prompt = prompt.format_prompt(query=query)
    output = model(format_prompt.to_string())
    format_output = parser.parse(output)
    # print("%d.用户问题: %s" % (i+1, query))
    print("提取故障设备、故障类型和查询时间范围如下：")
    print("1)故障设备: " + format_output.equipment_name.value)
    print("2)故障类型: " + format_output.failure_type.value)
    print("3)查询开始日期: " + format_output.time_period.start_day)
    print("4)查询结束日期: " + format_output.time_period.end_day)
    print("-----------------------------------\n")

