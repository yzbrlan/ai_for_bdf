from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_core.callbacks import BaseCallbackHandler

# 严格模式模板
strict_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    【角色限制】 
    你是一个专业MSC Nastran文档分析代理，请分析附件中的BDF文件并
    严格遵循以下规则：
    1. **仅输出文档原始片段**，不可总结/重组/解释/再加工
    2. 文档无相关内容时回复：『文档中未提及该内容』
    3. 禁止任何格式修改(包括标点/空格/段落格式)

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
    llm = OllamaLLM(
        base_url='http://localhost:11434',
        model="qwen2.5:14b", 
        # model="Qwen/Qwen2.5-72B-Instruct",
        # model="Qwen/QwQ-32B",
        temperature=0.01,
                )

    # 构建处理链
    chain = (
            {"doc_text": lambda x: "\n\n[文档块分隔]\n\n".join(chunks), "question": RunnablePassthrough()}
            | strict_prompt
            | llm
    )
    return chain

def count_tokens(result):
    print(f"本次消耗 {result.usage.total_tokens} 个 tokens")  # 包含输入+输出

class TokenCountCallbackHandler(BaseCallbackHandler):
    def handle_llm_end(self, response, **kwargs):
        count_tokens(response)

# 使用示例
if __name__ == "__main__":
    # qa_chain = strict_qa_system("data/cantilever_2014.1.bdf.txt")
    qa_chain = strict_qa_system("data/d173.txt")
    # 交互式问答
    # For cantilever case

    # For composite case
    grid_ids = [18,40,59,86,125]
    element_ids = [360,13,15,63,82]
    pids = [61, 1, 22]
    mids = [1]
    force_ids = [201]
    constraint_ids = [1]

    queries = [
        f"please return all grid cards with ID of {grid_ids} in plain text format in BULK data",
        # f"please return all element cards with EID of {element_ids} in plain text format in BULK data",
        # f"please return all property cards with PID of {pids} in plain text format in BULK data",
        # f"please return all material cards with MID of {mids} in plain text format in BULK data",
        # "please return all load cards including LOAD, FORCE with ID of {force_ids} in plain text format in BULK data",
        # "please return all constraint cards including SPC, MPC in plain text format in BULK data"
    ]
    print("文档问答系统（输入 exit 退出）")
    for query in queries:
        response = qa_chain.invoke(query, config={"callbacks": [TokenCountCallbackHandler()]})
        # breakpoint()
        print("【提问】", query)
        print("【响应】\n", response)
        # print("【token usage】\n", response.response_metadata['token_usage']['completion_tokens'])

