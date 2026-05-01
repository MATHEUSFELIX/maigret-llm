"""
Module 6 — Priorização Inteligente de Sites.

Dado o contexto da investigação e informações iniciais do alvo,
o LLM decide quais sites pesquisar primeiro — reduzindo ruído e tempo.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .llm import BaseLLMClient

_SYSTEM = """\
Você é um analista OSINT sênior especializado em inteligência digital.
Conheça bem as características de cada plataforma e como elas se relacionam
com diferentes perfis de investigação.
Responda SOMENTE com JSON válido."""

_PROMPT_TEMPLATE = """\
## Contexto da Investigação
{investigation_context}

## Informações Conhecidas do Alvo
{target_info}

## Sites Disponíveis no Maigret (amostra das categorias)
```json
{available_sites_json}
```

## Tarefa

Dado o contexto e o perfil do alvo, ordene os sites por relevância investigativa.

Princípios de priorização:
- **Investigação de fraude financeira**: priorizar exchanges crypto, fóruns de investimento,
  LinkedIn, Glassdoor, sites de freelance, Telegram, Discord
- **Investigação de abuso/assédio**: priorizar redes sociais, fóruns anônimos,
  plataformas de dating, Discord, Telegram
- **Investigação de identidade geral**: balancear entre redes sociais principais,
  plataformas profissionais e fóruns de nicho
- **Contexto tech/desenvolvedor**: priorizar GitHub, GitLab, Stack Overflow,
  HackerNews, dev.to, npm
- **Username parece de gamer**: priorizar Steam, Twitch, Discord, Reddit gaming subs

Responda com JSON:
```json
{{
  "reasoning": "Explicação da estratégia de priorização",
  "priority_tiers": {{
    "tier_1_critical": {{
      "sites": ["GitHub", "LinkedIn", "Twitter"],
      "rationale": "Sites com maior probabilidade de retorno imediato dado o contexto tech"
    }},
    "tier_2_high": {{
      "sites": ["Reddit", "HackerNews", "dev.to"],
      "rationale": "Complementares ao perfil técnico"
    }},
    "tier_3_medium": {{
      "sites": ["Facebook", "Instagram"],
      "rationale": "Vale verificar mas menor probabilidade"
    }},
    "tier_4_low": {{
      "sites": ["Pinterest", "TikTok"],
      "rationale": "Baixa relevância para este contexto"
    }},
    "skip": {{
      "sites": ["CookingForum", "KnittingPatterns"],
      "rationale": "Irrelevante para investigação de fraude financeira"
    }}
  }},
  "max_recommended": 100,
  "estimated_hit_rate": "Alta — perfil sugere presença ativa em plataformas tech"
}}
```
"""

# Common site categories for context
_SITE_CATEGORIES = {
    "professional": ["LinkedIn", "GitHub", "GitLab", "Behance", "Dribbble", "AngelList"],
    "social": ["Twitter", "Facebook", "Instagram", "TikTok", "Pinterest", "Snapchat"],
    "crypto_finance": ["Bitcointalk", "CryptoCompare", "CoinMarketCap", "Binance", "Etherscan"],
    "tech_dev": ["GitHub", "GitLab", "StackOverflow", "HackerNews", "dev.to", "npm"],
    "gaming": ["Steam", "Twitch", "Discord", "Xbox", "PSN", "Battlenet"],
    "forums": ["Reddit", "Quora", "HackerNews", "4chan", "Disqus"],
    "adult": ["OnlyFans", "FetLife", "AdultFriendFinder"],
    "dating": ["Tinder", "Bumble", "OkCupid", "Badoo"],
    "messaging": ["Telegram", "WhatsApp", "Signal", "Keybase"],
    "creative": ["DeviantArt", "ArtStation", "SoundCloud", "Bandcamp", "Wattpad"],
}


@dataclass
class PriorityTier:
    tier: str
    sites: list[str]
    rationale: str


@dataclass
class PrioritizationResult:
    reasoning: str
    tiers: list[PriorityTier]
    ordered_sites: list[str]  # Flat list ready to pass to Maigret
    max_recommended: int
    estimated_hit_rate: str
    cost_usd: float = 0.0
    model: str = ""

    def maigret_site_list(self, max_sites: Optional[int] = None) -> str:
        """Returns comma-separated site list for --top-sites or --site flag."""
        sites = self.ordered_sites[:max_sites] if max_sites else self.ordered_sites
        return ",".join(sites)


def prioritize_sites(
    llm: BaseLLMClient,
    investigation_context: str,
    target_info: Optional[dict[str, Any]] = None,
    available_sites: Optional[list[str]] = None,
) -> PrioritizationResult:
    """
    Prioriza sites para investigação dado o contexto.

    Args:
        llm: cliente LLM.
        investigation_context: Descrição do objetivo (ex: "fraude em criptomoedas").
        target_info: Informações conhecidas do alvo (username, nome, etc.).
        available_sites: Lista de sites disponíveis. Se None, usa categorias padrão.

    Returns:
        PrioritizationResult com sites ordenados por relevância.
    """
    if available_sites is None:
        all_sites = []
        for sites in _SITE_CATEGORIES.values():
            all_sites.extend(sites)
        available_sites = list(dict.fromkeys(all_sites))  # dedup preserving order

    # Sample to avoid too many tokens
    site_sample = available_sites[:200]
    sites_by_category = {
        cat: [s for s in sites if s in site_sample]
        for cat, sites in _SITE_CATEGORIES.items()
    }

    prompt = _PROMPT_TEMPLATE.format(
        investigation_context=investigation_context,
        target_info=json.dumps(target_info or {}, ensure_ascii=False, indent=2),
        available_sites_json=json.dumps(sites_by_category, ensure_ascii=False, indent=2),
    )

    response = llm.complete(prompt, system=_SYSTEM, max_tokens=2000)
    parsed = _parse_json(response.content)

    tiers_raw = parsed.get("priority_tiers", {})
    tiers: list[PriorityTier] = []
    ordered: list[str] = []

    for tier_name, tier_data in tiers_raw.items():
        if tier_name == "skip":
            continue
        sites = tier_data.get("sites", [])
        tiers.append(PriorityTier(
            tier=tier_name,
            sites=sites,
            rationale=tier_data.get("rationale", ""),
        ))
        for s in sites:
            if s not in ordered and s in available_sites:
                ordered.append(s)

    # Append any remaining sites not covered by tiers
    skip_sites = set(tiers_raw.get("skip", {}).get("sites", []))
    for s in available_sites:
        if s not in ordered and s not in skip_sites:
            ordered.append(s)

    return PrioritizationResult(
        reasoning=parsed.get("reasoning", ""),
        tiers=tiers,
        ordered_sites=ordered,
        max_recommended=int(parsed.get("max_recommended", 100)),
        estimated_hit_rate=parsed.get("estimated_hit_rate", ""),
        cost_usd=response.cost_usd,
        model=response.model,
    )


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
