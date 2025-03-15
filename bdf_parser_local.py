from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

# 严格模式模板
strict_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    【角色限制】 
    你是一个专业文档分析代理，严格遵循以下规则：
    1. **仅输出文档原始片段**，不可总结/重组/解释
    2. 文档无相关内容时回复：『文档中未提及该内容』
    3. 禁止任何格式修改(包括标点/空格/段落格式)
    4. 如果存在多个相关片段，全部用【原文引用块】展示

    【操作流程】
    1. 严谨匹配用户问题与文档语义
    2. 返回匹配片段的原始文本
    """),
    ("human", "目标文档：\n{doc_text}\n\n问题：{question}")
])

def read_file(file_path):
    """读取txt文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def strict_qa_system(file_path):
    # 加载文档
    loader = TextLoader(file_path)
    doc = loader.load()[0].page_content

    # 分割文本块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(doc)

    # 核心模型设置 ✅
    llm = OllamaLLM(base_url='http://localhost:11434',
    model="qwen2.5:14b",    # 使用正确的Ollama模型名称
    temperature=0.2,
    num_predict=512      # 控制生成长度)
                )

    # 构建处理链
    chain = (
            {"doc_text": lambda x: "\n\n[文档块分隔]\n\n".join(chunks), "question": RunnablePassthrough()}
            | strict_prompt
            | llm
    )
    return chain


# 使用示例
if __name__ == "__main__":
    qa_chain = strict_qa_system("data/cantilever_2014.1.bdf.txt")

    # 交互式问答
    print("文档问答系统（输入 exit 退出）")
    while True:
        query = input("\n请输入问题：")
        if query.lower() == "exit":
            break
        if not query.strip():
            continue

        response = qa_chain.invoke(query)
        # breakpoint()
        print("【提问】", query)
        # print("【响应】\n", response.content)
        print("【响应】\n", response)

