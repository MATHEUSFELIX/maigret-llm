"""
Module 5 — Parser Adaptativo de Páginas de Perfil.

Usa LLM como parser universal de HTML de perfis — resiliente a redesigns.
Aplicar seletivamente em sites de alto valor onde o parser regex quebrou.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from .llm import BaseLLMClient

_SYSTEM = """\
Você é um expert em extração de dados de páginas web.
Sua tarefa é extrair campos estruturados de HTML de páginas de perfil.
Responda SOMENTE com JSON válido — sem texto adicional, sem markdown."""

_PROMPT_TEMPLATE = """\
## URL do Perfil
{url}

## Plataforma
{platform}

## HTML da Página (primeiros {html_size} caracteres)
```html
{html}
```

## Tarefa

Extraia todos os campos disponíveis deste perfil em formato JSON estruturado.
Se um campo não estiver presente, omita-o (não use null).

Formato esperado:
```json
{{
  "username": "johndoe",
  "display_name": "John Doe",
  "bio": "Software engineer @ ...",
  "location": "São Paulo, Brasil",
  "website": "https://johndoe.dev",
  "email": "john@example.com",
  "phone": "+55 11 9...",
  "created_at": "2018-03-15",
  "last_active": "2024-01-20",
  "followers": 1234,
  "following": 567,
  "posts_count": 89,
  "verified": false,
  "avatar_url": "https://...",
  "external_links": ["https://...", "https://..."],
  "unique_ids": {{"twitter_id": "123456789", "user_id": "abc123"}},
  "additional_fields": {{
    "campo_especifico_da_plataforma": "valor"
  }}
}}
```

Extraia o máximo de informação possível. Priorize:
1. Identificadores únicos (IDs, emails, telefones)
2. Links externos (podem levar a outros perfis)
3. Bio e localização
4. Metadados temporais (criação, última atividade)
"""

_HTML_SIZE = 12_000  # chars sent to LLM — balance cost vs completeness


@dataclass
class ParsedProfile:
    url: str
    platform: str
    fields: dict[str, Any]
    raw_html_size: int = 0
    cost_usd: float = 0.0
    model: str = ""
    parse_error: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        return self.fields.get(key, default)

    @property
    def unique_ids(self) -> dict[str, str]:
        return self.fields.get("unique_ids", {})

    @property
    def external_links(self) -> list[str]:
        return self.fields.get("external_links", [])


def parse_profile(
    url: str,
    llm: BaseLLMClient,
    platform: str = "",
    html: Optional[str] = None,
    html_size: int = _HTML_SIZE,
) -> ParsedProfile:
    """
    Extrai campos estruturados de uma página de perfil via LLM.

    Args:
        url: URL completa do perfil.
        llm: cliente LLM.
        platform: Nome da plataforma (para contexto).
        html: HTML já buscado (se None, busca automaticamente).
        html_size: Quantos caracteres de HTML enviar ao LLM.

    Returns:
        ParsedProfile com todos os campos extraídos.
    """
    if html is None:
        html = _fetch_html(url)

    raw_size = len(html)
    html_sample = html[:html_size]

    if not platform:
        # Infer platform from URL
        match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        platform = match.group(1) if match else url

    prompt = _PROMPT_TEMPLATE.format(
        url=url,
        platform=platform,
        html=html_sample,
        html_size=html_size,
    )

    try:
        response = llm.complete(prompt, system=_SYSTEM, max_tokens=2000)
        parsed = _parse_json(response.content)

        return ParsedProfile(
            url=url,
            platform=platform,
            fields=parsed,
            raw_html_size=raw_size,
            cost_usd=response.cost_usd,
            model=response.model,
        )
    except Exception as e:
        return ParsedProfile(
            url=url,
            platform=platform,
            fields={},
            raw_html_size=raw_size,
            parse_error=str(e),
        )


def parse_profiles_from_maigret(
    maigret_results: dict[str, Any],
    llm: BaseLLMClient,
    high_value_sites: Optional[list[str]] = None,
    max_sites: int = 10,
) -> list[ParsedProfile]:
    """
    Extrai perfis detalhados dos sites encontrados pelo Maigret.

    Aplica o parser LLM seletivamente — apenas nos sites de maior Valor
    ou onde o parser regex do Maigret não extraiu dados suficientes.

    Args:
        maigret_results: Resultado bruto do Maigret.
        llm: cliente LLM.
        high_value_sites: Lista de sites prioritários (ex: ["LinkedIn", "GitHub"]).
                          Se None, usa os primeiros max_sites encontrados.
        max_sites: Número máximo de sites para parsear.
    """
    results: list[ParsedProfile] = []

    for site, data in maigret_results.items():
        if not isinstance(data, dict):
            continue

        # Only process found accounts
        status = data.get("status", {})
        if isinstance(status, dict) and status.get("id") != 0:
            continue

        # Filter by high_value_sites if provided
        if high_value_sites and site not in high_value_sites:
            continue

        if len(results) >= max_sites:
            break

        url = data.get("url_user") or data.get("url", "")
        if not url or "{}" in url:
            continue

        print(f"[adaptive_parser] Parseando: {site} ({url})")
        profile = parse_profile(url=url, llm=llm, platform=site)
        results.append(profile)

    return results


def _fetch_html(url: str, timeout: int = 15) -> str:
    """Fetch HTML with a browser user agent."""
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except URLError as e:
        return f"<!-- fetch error: {e} -->"


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}
