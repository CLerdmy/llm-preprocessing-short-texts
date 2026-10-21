"""Клиент GigaChat"""

import base64
import os
import time
import uuid

import requests
import urllib3
from dotenv import load_dotenv

from src.paths import ENV
from src.llm.base import BaseLLM

load_dotenv(ENV)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
API_URL = "https://api.giga.chat/v1/chat/completions"


class GigaChat(BaseLLM):

    def __init__(self, model: str = "GigaChat-2", auth_key: str | None = None, scope: str | None = None) -> None:
        self.model = model
        self.scope = scope or os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.auth_key = auth_key or self._auth_key_from_env()
        self._token: str | None = None
        self._expires: float = 0.0

    @staticmethod
    def _auth_key_from_env() -> str:
        key = os.getenv("GIGACHAT_AUTHORIZATION_KEY")
        if key:
            return key
        client_id = os.getenv("GIGACHAT_CLIENT_ID")
        secret = os.getenv("GIGACHAT_CLIENT_SECRET")
        if client_id and secret:
            return base64.b64encode(f"{client_id}:{secret}".encode()).decode()
        raise RuntimeError("в .env нет GIGACHAT_AUTHORIZATION_KEY или пары GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET")

    def token(self) -> str:
        if self._token and time.time() < self._expires - 60:
            return self._token
        response = requests.post(
            OAUTH_URL,
            headers={"Authorization": f"Basic {self.auth_key}", "RqUID": str(uuid.uuid4()), "Content-Type": "application/x-www-form-urlencoded"},
            data={"scope": self.scope}, verify=False, timeout=30)
        response.raise_for_status()
        data = response.json()
        self._token = data["access_token"]
        self._expires = data.get("expires_at", 0) / 1000 or time.time() + 1800
        return self._token

    def ask(self, text: str, prompt: str | None = None, temperature: float | None = None, timeout: int = 180) -> dict:
        messages = ([{"role": "system", "content": prompt}] if prompt else []) + [{"role": "user", "content": text}]
        payload = {"model": self.model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature

        started = time.time()
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {self.token()}"},
            json=payload, verify=False, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return {"text": data["choices"][0]["message"]["content"], "tokens": data["usage"]["total_tokens"], "elapsed": round(time.time() - started, 2)}
