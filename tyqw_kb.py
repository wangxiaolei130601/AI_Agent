# 1.pip install tqdm
# 2.pip install unstructured
# 3.pip install "unstructured[xlsx]"
# 4.pip install cmake
# 5.pip install "unstructured[pdf]"
# 6.python3.12无法安装chromadb，降低到python3.9
# 7.windows11无法安装chromadb，先下载并安装VS生成工具，重启电脑。https://stackoverflow.com/questions/73969269/error-could-not-build-wheels-for-hnswlib-which-is-required-to-install-pyprojec/76245995#76245995
# 8.pip install chromadb
# 9.python -m pip install grpcio --ignore-installed
# 10.在project目录下创建目录/LangChain/KnowledgeBase/，把知识库文件pdf复制到目录中


# 引入环境变量：通义千问大模型API_KEY和向量检索服务API_KEY
import os
os.environ["DASHSCOPE_API_KEY"] = "sk-ecdbc4a7a2b140c69b04c6d5953a11d6"
os.environ["DASHVECTOR_API_KEY"] = "sk-XN91N5YrendXVzFrihpgafxpTfbIQ9ABFD0E2BC5811EE81772A30DF8C6901"

from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import DashScopeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Tongyi
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

# 从本地读取相关数据
loader = DirectoryLoader(
     './LangChain/KnowledgeBase/', glob='**/*.pdf', show_progress=True
)
docs = loader.load()

# 将文本进行分割
text_splitter = CharacterTextSplitter(
     chunk_size=1000,
     chunk_overlap=0
)
docs_split = text_splitter.split_documents(docs)

# 初始化OpenAI Embeddings
embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=os.environ["DASHSCOPE_API_KEY"])

# 将数据存入DashVector向量存储
vector_store_path = "./LangChain/vector_store"
vector_store = Chroma.from_documents(docs_split, embeddings)
#vector_store = Chroma.from_documents(documents=docs_split,
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

# 初始化大模型
llm = Tongyi(temperature=0.0, max_tokens=2048)

# 初始化问答链
qa = ConversationalRetrievalChain.from_llm(llm, retriever, condense_question_prompt=prompt)

chat_history = []
while True:
  question = input(' 问题：')
  # 开始发送问题chat_history为必须参数，用于存储历史消息
  result = qa({'question': question, 'chat_history': chat_history})
  chat_history.append((question, result['answer']))
  print(result['answer'])