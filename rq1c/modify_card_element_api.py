import os
import sys
import time

import json5

from rq1c.file_util import logger_config

sys.path.append("./")
sys.path.append("../..")
sys.path.append("../../")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler

# 严格模式模板
strict_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    【角色限制】 
    你是一个专业MSC Nastran文档分析代理，请分析附件中的BDF文件并
    严格遵循以下规则：
    1. **仅输出文档原始片段**，不可总结/重组/解释/再加工
    2. 文档无相关内容时回复：『文档中未提及该内容』
    3. 问题不符合MSC Nastran文档格式回复：『问题不正确』
    4. 禁止任何格式修改(包括标点/空格/段落格式)

    【操作流程】
    1. 校验用户问题是否正确，符合MSC Nastran文档格式
    2. 严谨匹配用户问题与文档语义
    """),
    ("human", "目标文档：\n{doc_text}\n\n问题：{question}")
])


def read_file(file_path):
    """读取txt文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def remote_strict_qa_system(bdf_file_path, remote_model_param_dict):
    # 加载文档
    loader = TextLoader(bdf_file_path)
    doc = loader.load()[0].page_content

    # 分割文本块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(doc)

    # 核心模型设置 ✅
    llm = ChatOpenAI(
        api_key=remote_model_param_dict["api_key"],
        base_url=remote_model_param_dict["base_url"],
        model=remote_model_param_dict["model"],
        temperature=remote_model_param_dict["temperature"],
        # max_tokens=1024,
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
    # 初始化日志格式
    date_str = time.strftime('%Y-%m-%d', time.localtime())
    time_str = time.strftime('%H:%M:%S', time.localtime())
    result_path = os.path.join(os.getcwd(), f"result/{date_str}")
    logger = logger_config(result_path, f"{time_str}.txt")

    logger.info(f"-------------------------start new experiment-------------------------------")

    # 初始化模型参数，json5字符串
    with open("package.json", encoding="utf-8") as f1:
        params = json5.load(f1)
        logger.info(f"current package.json param is {json5.dumps(params, indent=4)}")

    # 不同bdf的参数：bdf路径，对应的问题列表
    bdf_param_list = params["bdf_param_list"]
    for bdf_param_dict in bdf_param_list:
        bdf_file_name = bdf_param_dict["bdf_file_name"]
        root_path = os.path.abspath(os.path.dirname(os.getcwd()))
        bdf_file_path = os.path.join(root_path, "data", bdf_file_name)
        logger.info(f"-----------------current bdf file is 【{bdf_file_name}】")

        question_param_list = bdf_param_dict["question_param_list"]

        count = 1
        for question in question_param_list:
            # TODO:组装问题
            question = question
            logger.info("")
            logger.info(f"bdf【{bdf_file_name}】问题【{count}】is【 {question} 】")

            # 遍历不同大模型
            remote_model_list = params["remote_model_list"]
            for remote_model_param_dict in remote_model_list:
                logger.info("")
                logger.info(f"-----------------current model is 【{remote_model_param_dict['model']}】")

                qa_chain = remote_strict_qa_system(bdf_file_path, remote_model_param_dict)

                logger.info("")
                logger.info(f"bdf【{bdf_file_name}】问题【{count}】is【 {question} 】")
                start_time = time.time()
                response = qa_chain.invoke(question, config={"callbacks": [TokenCountCallbackHandler()]})

                logger.info(
                    f"token usage =【{response.response_metadata['token_usage']['completion_tokens']}】,time usage =【{start_time - start_time}】")
                logger.info(f"model【{remote_model_param_dict['model']}】响应【{count}】is \n{response.content}")

            count += 1
