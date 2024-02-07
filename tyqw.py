import os
os.environ["DASHSCOPE_API_KEY"] = "sk-ecdbc4a7a2b140c69b04c6d5953a11d6"

#意图识别
from langchain.llms import Tongyi
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

template = """Question: {question}

Answer: 用户提出的问题是下列两种问题其中之一：1.询问当前发生的设备问题的故障定位和处理方法；2.询问历史发生过的类似设备故障。理解用户意图，答案返回1或2，不要自由发挥."""

prompt = PromptTemplate(
    template=template,
    input_variables=["question"])

print(prompt)

llm = Tongyi(temperature=0)

llm_chain = LLMChain(prompt=prompt, llm=llm)

topic = "以前出现过的脉冲阀问题有哪些？"

res = llm_chain.run(topic)

print(res)

