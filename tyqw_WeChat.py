import os

os.environ["DASHSCOPE_API_KEY"] = "sk-ecdbc4a7a2b140c69b04c6d5953a11d6"

from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
import json

# 1. 初始化大模型
llm = Tongyi(temperature=0.0)

# 2. 初始化设备故障智能助理Failure Supervisor
# 定义Failure Supervisor可以调用的tool agent
tool_agents = ["Search_Same_Type_Failures", "Search_Same_Equipment_Failures", "Search_OM_Knowledge_Base"]
tool_intention = {
    "Search_Same_Type_Failures": "查看同类故障",
    "Search_Same_Equipment_Failures": "查看同设备故障",
    "Search_OM_Knowledge_Base": "查看故障维修方法"
}
system_prompt = (
    "Given a user request, respond with one of the following tools. Each tool will perform a task."
    # "Here are the tools and their descriptions you can choose:\n{tools_desc}"
    "Here are the tools and their capabilities:"
    "1. Search_Same_Type_Failures: Search database then return the failures of same type."
    "2. Search_Same_Equipment_Failures: Search database then return the failures of same equipment."
    "3. Search_OM_Knowledge_Base: Search knowledge base then answer how to diagnose or repair any equipment failure."
)
output_def = {
    "name_for_model": "supervisor",
    "description_for_model": "Select the tool responding with user request."
                             + " Format the arguments as a JSON object.",
    "parameters": [
        {
            "name": "next",
            "description": "the next tool to respond with user request.",
            "required": True,
            "type": "string",
            "enum": tool_agents
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
            " Only select one of: {tool_agents}",
        ),
    ],
).partial(tool_agents=str(tool_agents), format_instructions=output_format)

supervisor = LLMChain(prompt=prompt, llm=llm, output_key="next")

# 3. 建立数据库连接
import pymysql
from pymysql.cursors import Cursor


db = pymysql.connect(host='127.0.0.1',
                     user='root',
                     password='123456',
                     database='equipments')
table_name = "equipments"


# 4. 初始化本地知识库
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import DashScopeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate


knowledge_base_path = "./LangChain/KnowledgeBase/"
vector_store_path = "./LangChain/vector_store"
chat_history = []


def initialize_om_knowledge_base(llm: Tongyi, kb_path: str, ):
    # 从本地读取相关数据
    loader = DirectoryLoader(kb_path, glob='**/*.pdf', show_progress=True)
    docs = loader.load()

    # 将文本进行分割
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)

    # 初始化OpenAI Embeddings
    embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=os.environ["DASHSCOPE_API_KEY"])

    # 向量化
    vector_store = Chroma.from_documents(docs_split, embeddings)

    # 将数据存入DashVector向量存储，不用每次重新向量化
    # vector_store_path = vs_path
    # vector_store = Chroma.from_documents(documents=docs_split,
    #                                         embedding=embeddings,
    #                                         persist_directory=vector_store_path)

    # 初始化检索器，使用向量存储
    retriever = vector_store.as_retriever()

    system_template = """
    Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answering these questions in Chinese.
    -----------
    {question}
    -----------
    {chat_history}
    """

    # 构建初始消息列表
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template('{question}')
    ]

    # 初始化Prompt对象
    prompt = ChatPromptTemplate.from_messages(messages)

    # 初始化问答链
    qa = ConversationalRetrievalChain.from_llm(llm, retriever, condense_question_prompt=prompt)

    return qa


qa = initialize_om_knowledge_base(llm, knowledge_base_path)

# 查看同类故障
def search_same_type_failures(db, table: str, type: str):
    sql = "SELECT * FROM %s WHERE type = '%s' " % (table, type)
    cursor = db.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()

    return result


# 查看同设备故障
def search_same_equipment_failures(db, table: str, equipment: str):
    sql = "SELECT * FROM %s WHERE equipment = '%s' " % (table, equipment)
    cursor = db.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()

    return result


# 查看故障诊断与维修知识
def search_om_knowledge_base(qa, question: str, chat_history: list):
    result = qa({'question': question, 'chat_history': chat_history})
    # chat_history.append((question, result['answer']))
    return result


# demo
# 直接从后端获取的结构化信息，无需大模型提取
equipment_name = "中碎筛分除尘器"
failure_type = "差压过高"
inputs = [
    "差压过高故障还出现过哪些？",
    "中碎筛分除尘器还出现过什么故障？",
    "差压过高原因怎么排查？",
    "输灰不畅怎么修理？"
]
for request in inputs:
    respond = supervisor.invoke({"input": request})
    message = respond["input"]
    tool_agent = respond["next"]
    print("用户消息: " + request)
    print("用户意图: " + tool_intention[tool_agent])
    if tool_agent == "Search_Same_Type_Failures":
        result = search_same_type_failures(db, table_name, failure_type)
        for i in range(len(result)):
            res = result[i]
            create_time = res[4].strftime("%Y-%m-%d %H:%M:%S")
            print("%d: %s\t%s" % (i + 1, res[1], res[4]))
        print("---------------------------------")
    elif tool_agent == "Search_Same_Equipment_Failures":
        result = search_same_equipment_failures(db, table_name, equipment_name)
        for i in range(len(result)):
            res = result[i]
            create_time = res[4].strftime("%Y-%m-%d %H:%M:%S")
            print("%d: %s\t%s" % (i + 1, res[1], res[4]))
        print("---------------------------------")
    elif tool_agent == "Search_OM_Knowledge_Base":
        result = search_om_knowledge_base(qa, message, chat_history)
        print(result["answer"])
        print("---------------------------------")

db.close()
