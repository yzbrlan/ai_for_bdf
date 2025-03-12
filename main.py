
from openai import OpenAI

client = OpenAI(api_key="YOUR_KEY", base_url="https://api.siliconflow.cn/v1")

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about recursion in programming."}
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=True
)
# 逐步接收并处理响应
for chunk in response:
    print(chunk.choices[0].delta.content)