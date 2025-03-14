from openai import OpenAI
import os

def read_file(file_path):
    """读取txt文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def qa_from_file(file_path, question):
    """
    基于文件内容回答问题的函数
    :param file_path: txt文件路径
    :param question: 用户问题
    :return: 模型生成的答案
    """
    context = read_file(file_path)
    client = OpenAI(api_key="sk-oscozltnsdfkayszykrkkedaccqdkroauugqfavifncvugzg",
                    base_url="https://api.siliconflow.cn/v1")

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "system", "content": "You are familiar with MSC nastran and the BDF input files."},
            {"role": "user", "content": f"Document content:\n{context}\n\nQuestion to answer: {question}"}
        ],
        temperature=0,
        max_tokens=1024,
        stream=False
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # 配置参数
    TXT_FILE = "data/cantilever_2014.1.bdf.txt"  # 修改为你的文件路径

    # 交互式问答
    print("文档问答系统（输入 exit 退出）")
    while True:
        query = input("\n请输入问题：")
        if query.lower() == "exit":
            break
        if not query.strip():
            continue

        answer = qa_from_file(TXT_FILE, query)
        print(f"\n答案：{answer}")
