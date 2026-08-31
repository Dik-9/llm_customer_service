# -*- coding: utf-8 -*-
"""直接测试 DashScope API 连通性（绕过 LangChain）"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env
load_dotenv(Path(__file__).parent / ".env")

# ====== 测试1: OpenAI 兼容模式 ======
print("=" * 50)
print("测试1: OpenAI 兼容模式 (base_url + openai client)")
print("=" * 50)

api_key = os.getenv("DASHSCOPE_API_KEY")
print(f"API Key: {'已加载' if api_key else '未找到'}")

from openai import OpenAI

client = OpenAI(
    api_key=api_key,
    base_url="https://llm-xcmc3bcdk6yz5n2b.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

try:
    resp = client.chat.completions.create(
        model="qwen3.6-plus",
        messages=[{"role": "user", "content": "你好，请说一个字"}],
        max_tokens=50,
        temperature=0,
    )
    content = resp.choices[0].message.content
    print(f"✅ 成功! 响应: {content}")
    print(f"   model: {resp.model}")
    print(f"   usage: {resp.usage}")
except Exception as e:
    print(f"❌ 失败: {type(e).__name__}: {e}")


# ====== 测试2: DashScope 原生模式 ======
print()
print("=" * 50)
print("测试2: DashScope 原生 SDK")
print("=" * 50)

try:
    import dashscope
    from dashscope import Generation

    resp = Generation.call(
        model="qwen3.6-plus",
        messages=[{"role": "user", "content": "你好，请说一个字"}],
        api_key=api_key,
        max_tokens=50,
        temperature=0,
        result_format="message",
    )
    if resp.status_code == 200:
        print(f"✅ 成功! 响应: {resp.output.choices[0].message.content}")
    else:
        print(f"❌ 失败: code={resp.code}, message={resp.message}")
except ImportError:
    print("⚠️  dashscope SDK 未安装")
except Exception as e:
    print(f"❌ 失败: {type(e).__name__}: {e}")


# ====== 测试3: 用不同模型名测试兼容模式 ======
print()
print("=" * 50)
print("测试3: 兼容模式 + qwen-plus 模型")
print("=" * 50)

try:
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "你好，请说一个字"}],
        max_tokens=50,
        temperature=0,
    )
    content = resp.choices[0].message.content
    print(f"✅ 成功! 响应: {content}")
except Exception as e:
    print(f"❌ 失败: {type(e).__name__}: {e}")
