from openai import OpenAI

from .config import get_settings


class LLMConfigurationError(RuntimeError):
    pass


PLACEHOLDER_KEYS = {"sk-your-api-key", "your-api-key", "your_api_key", "replace-me"}


def _valid_api_key() -> str | None:
    return _clean_api_key(get_settings().llm_api_key)


def _clean_api_key(api_key: str | None) -> str | None:
    if not api_key:
        return None
    stripped = api_key.strip()
    if not stripped or stripped.lower() in PLACEHOLDER_KEYS:
        return None
    return stripped


def llm_configured() -> bool:
    return bool(_valid_api_key())


def complete(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    settings = get_settings()
    api_key = _valid_api_key()
    if not api_key:
        raise LLMConfigurationError("LLM_API_KEY is missing. Configure .env before using LLM features.")

    client = OpenAI(api_key=api_key, base_url=settings.llm_base_url)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned an empty response.")
    return content.strip()


def vision_complete(system_prompt: str, prompt: str, images: list[dict], temperature: float = 0.0) -> str:
    settings = get_settings()
    api_key = _valid_api_key()
    if not api_key:
        raise LLMConfigurationError("LLM_API_KEY is missing. Configure .env before using Kimi k2.6 visual judgement.")

    content = [{"type": "text", "text": prompt}]
    for image in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image["data_url"],
                    "detail": image.get("detail", "low"),
                },
            }
        )
    client = OpenAI(api_key=api_key, base_url=settings.llm_base_url)
    response = client.chat.completions.create(
        model="kimi-k2.6",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=temperature,
    )
    result = response.choices[0].message.content
    if not result:
        raise RuntimeError("Kimi k2.6 returned an empty visual judgement response.")
    return result.strip()


def stream_complete(system_prompt: str, user_prompt: str, temperature: float = 0.2):
    settings = get_settings()
    api_key = _valid_api_key()
    if not api_key:
        raise LLMConfigurationError("LLM_API_KEY is missing. Configure .env before using LLM features.")

    client = OpenAI(api_key=api_key, base_url=settings.llm_base_url)
    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta
