# from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(base_url='http://localhost:11434',
    model="qwen2:7b",    # 使用正确的Ollama模型名称
    temperature=0.7,
    num_predict=512      # 控制生成长度)
)
