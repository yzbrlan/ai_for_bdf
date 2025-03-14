from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = "sk-oscozltnsdfkayszykrkkedaccqdkroauugqfavifncvugzg"

# 1. 加载文档
loader = TextLoader("your_file.txt", encoding="utf-8")
documents = loader.load()

# 2. 文档分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个文本块的大小
    chunk_overlap=50  # 块之间的重叠量
)
texts = text_splitter.split_documents(documents)

# 3. 创建向量数据库
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./db"  # 向量数据库存储路径
)

# 4. 创建问答链
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

prompt_template = """根据以下上下文信息回答问题：
{context}

问题：{question}
请只基于提供的上下文回答，如果不知道答案就说不知道。"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# 使用示例
question = "你的问题内容"
result = qa_chain.invoke({"query": question})

print("答案：", result["result"])
print("参考内容：")
for doc in result["source_documents"]:
    print(doc.page_content)
