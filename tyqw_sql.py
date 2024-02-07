# 1.报错找不到SQLDatabaseChain，需要安装pip install langchain-experimental
# 2.pip install cryptography
# 3.langchain写的sql查询提示词模板有问题：生成的sql查询语句不要用""包围
#   在langchain.chains.sql_database.prompt.py中更新提示词模板，增加以下几句话：
#     The SQL query should be outputted plainly, do not surround it in quotes or anything else.
#     Only use the tables listed below.
#
#     {table_info}
#
#     Question: {input}
#   参考https://github.com/langchain-ai/langchain/issues/2027


# 引入环境变量：通义千问大模型API_KEY
import os
os.environ["DASHSCOPE_API_KEY"] = "sk-ecdbc4a7a2b140c69b04c6d5953a11d6"

# 建立数据库连接
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
db = SQLDatabase.from_uri("mysql+pymysql://root:123456@127.0.0.1/testdb")

# 建立大模型
from langchain_community.llms import Tongyi
llm = Tongyi(temperature=0)

# 设置sql agent提示词
from langchain.prompts.prompt import PromptTemplate
_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
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
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)

# 建立数据库Chain
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True, prompt=PROMPT)

# 测试自然语言转SQL查询
db_chain.run("How many students are there?")
"""
db_chain.run("What are the students names?")
db_chain.run("What's the average score of them?")
db_chain.run("What's the average score of them, excluding the zero score?")
db_chain.run("Who got zero score?")
db_chain.run("Who got zero score? Why?")
db_chain.run("Who got zero score? Show me her parent's contact information.")
"""