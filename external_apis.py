from typing import Optional

import httpx
from astrbot.api import logger


async def translate_text(
    text: str,
    http_client: httpx.AsyncClient,
    api_config: dict,
    system_prompt_override: Optional[str] = None,
) -> Optional[str]:
    """
    使用配置的API进行翻译。
    :param text: 需要翻译的文本（或用于生成提示词的原始文本）。
    :param http_client: httpx 异步客户端实例。
    :param api_config: 插件配置中的 'translation_api' 部分。
    :param system_prompt_override: 可选，用于覆盖配置中的默认系统提示词。
    """
    base_url = api_config.get("base_url")
    api_key = api_config.get("api_key")
    model = api_config.get("model", "gpt-3.5-turbo")
    api_format = api_config.get("api_format", "openai")

    # 优先使用覆盖的提示词，否则使用配置的通用提示词
    system_prompt = system_prompt_override or api_config.get(
        "prompt", "You are a translation assistant."
    )

    if not all([base_url, api_key]):
        logger.error("翻译API配置不完整 (base_url, api_key)。")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        if api_format == "openai":
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            }
            endpoint_url = f"{base_url.strip('/')}/chat/completions"
        elif api_format == "gemini":
            payload = {
                "contents": [{"parts": [{"text": text}]}],
                "systemInstruction": {"parts": [{"text": system_prompt}]},
            }
            endpoint_url = f"{base_url.strip('/')}/v1beta/models/{model}:generateContent?key={api_key}"
            headers.pop("Authorization", None)
        else:
            logger.error(f"不支持的API格式: {api_format}")
            return None

        response = await http_client.post(
            endpoint_url, headers=headers, json=payload, timeout=120.0
        )
        response.raise_for_status()
        data = response.json()

        if api_format == "openai":
            return data["choices"][0]["message"]["content"]
        elif api_format == "gemini":
            return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        logger.error(f"翻译请求失败: {e}\n响应: {getattr(response, 'text', 'N/A')}")
        return None

    return None
