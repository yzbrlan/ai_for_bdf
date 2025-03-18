
from openai import OpenAI

client = OpenAI(api_key="sk-zcpmjwctlnkthagcorztornyziblyxvkjireovscqsymjqyn", base_url="https://api.siliconflow.cn/v1")

#attach external file
# file = client.files.create(
#     file=open("test.bdf", "rb"),
#     purpose="assistants"
# )

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",
    messages=[
        {"role": "system", "content": "You are familiar with MSC nastran and the BDF input files."},
        {"role": "user", "content": "Write a haiku about recursion in programming."}
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=True
)
# 逐步接收并处理响应
for chunk in response:
    breakpoint()
    print(chunk.choices[0].delta.content)