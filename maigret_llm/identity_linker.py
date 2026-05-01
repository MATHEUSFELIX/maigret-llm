"""
Module 4 — Detecção de Identidade Cruzada (Cross-Platform Identity Linking).

Analisa semanticamente dois perfis de plataformas diferentes e calcula
a probabilidade de pertencerem à mesma pessoa.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .llm import BaseLLMClient

_SYSTEM = """\
Você é um especialista em análise forense digital e investigação de identidade.
Sua análise combina técnicas de estilometria, análise comportamental e
correlação de metadados para detectar co-identidade entre perfis online.
Responda SOMENTE com JSON válido."""

_PROMPT_TEMPLATE = """\
## Perfil A — {platform_a}
```json
{profile_a_json}
```

## Perfil B — {platform_b}
```json
{profile_b_json}
```

## Tarefa: Análise de Co-Identidade

Analise os dois perfis e determine a probabilidade de pertencerem à mesma pessoa.

Considere TODOS os vetores disponíveis:

**Vetores de alta confiança:**
- Mesmo email ou número de telefone
- Mesmo ID único (ex: user_id compartilhado)
- Foto de perfil idêntica ou similar
- Bio idêntica ou muito similar

**Vetores de média confiança:**
- Username com padrão similar (mesmas iniciais, mesmos números)
- Localização compatível
- Mesmo nicho de interesse
- Vocabulário e estilo de escrita similares (estilometria básica)
- Período de atividade compatível

**Vetores de baixa confiança:**
- Mesma faixa etária estimada
- Interesses em comum genéricos
- Horário de atividade compatível com mesmo fuso horário

**Penalizações:**
- Localização contraditória
- Idiomas diferentes sem explicação
- Datas de criação muito distantes sem explicação

Responda com JSON:
```json
{{
  "score": 78,
  "verdict": "provável mesma pessoa",
  "confidence_label": "alta | média | baixa",
  "evidence": [
    {{
      "vector": "username similar",
      "weight": "média",
      "detail": "Perfil A usa 'jp.almeida', Perfil B usa 'jpalmeida90' — Mesmo padrão com ano adicionado"
    }}
  ],
  "contradictions": [
    {{
      "vector": "localização",
      "detail": "Perfil A diz São Paulo, Perfil B diz Lisboa — pode ser migração ou VPN"
    }}
  ],
  "summary": "Parágrafo explicando o raciocínio completo",
  "recommended_action": "O que investigar a seguir para confirmar ou refutar"
}}
```

Score: 0 = definitivamente pessoas diferentes, 100 = certeza de mesma pessoa.
"""


@dataclass
class EvidenceItem:
    vector: str
    weight: str
    detail: str


@dataclass
class LinkingResult:
    score: int
    verdict: str
    confidence_label: str
    evidence: list[EvidenceItem]
    contradictions: list[EvidenceItem]
    summary: str
    recommended_action: str
    platform_a: str = ""
    platform_b: str = ""
    cost_usd: float = 0.0
    model: str = ""

    @property
    def is_likely_same(self) -> bool:
        return self.score >= 65


def score_same_person(
    profile_a: dict[str, Any],
    profile_b: dict[str, Any],
    llm: BaseLLMClient,
    platform_a: str = "Plataforma A",
    platform_b: str = "Plataforma B",
) -> LinkingResult:
    """
    Calcula a probabilidade de dois perfis pertencerem à mesma pessoa.

    Args:
        profile_a: Dados do perfil A (dict com campos do Maigret).
        profile_b: Dados do perfil B.
        llm: cliente LLM.
        platform_a: Nome da plataforma A (para contexto).
        platform_b: Nome da plataforma B.

    Returns:
        LinkingResult com score 0-100 e evidências detalhadas.
    """
    prompt = _PROMPT_TEMPLATE.format(
        platform_a=platform_a,
        platform_b=platform_b,
        profile_a_json=json.dumps(profile_a, ensure_ascii=False, indent=2)[:4000],
        profile_b_json=json.dumps(profile_b, ensure_ascii=False, indent=2)[:4000],
    )

    response = llm.complete(prompt, system=_SYSTEM, max_tokens=2000)
    parsed = _parse_json(response.content)

    evidence = [
        EvidenceItem(
            vector=e.get("vector", ""),
            weight=e.get("weight", ""),
            detail=e.get("detail", ""),
        )
        for e in parsed.get("evidence", [])
    ]

    contradictions = [
        EvidenceItem(
            vector=c.get("vector", ""),
            weight=c.get("weight", "baixa"),
            detail=c.get("detail", ""),
        )
        for c in parsed.get("contradictions", [])
    ]

    return LinkingResult(
        score=int(parsed.get("score", 0)),
        verdict=parsed.get("verdict", "indeterminado"),
        confidence_label=parsed.get("confidence_label", "baixa"),
        evidence=evidence,
        contradictions=contradictions,
        summary=parsed.get("summary", ""),
        recommended_action=parsed.get("recommended_action", ""),
        platform_a=platform_a,
        platform_b=platform_b,
        cost_usd=response.cost_usd,
        model=response.model,
    )


def link_all_profiles(
    maigret_results: dict[str, Any],
    llm: BaseLLMClient,
    min_score_to_report: int = 50,
) -> list[LinkingResult]:
    """
    Compara todos os pares de perfis encontrados pelo Maigret.

    Útil quando usernames diferentes foram usados na mesma investigação
    e queremos descobrir quais pertencem à mesma pessoa.

    Args:
        maigret_results: Resultado bruto do Maigret com múltiplos usernames.
        llm: cliente LLM.
        min_score_to_report: Só retorna pares com score acima deste valor.
    """
    # Extract found profiles
    profiles = {}
    for site, data in maigret_results.items():
        if isinstance(data, dict) and data.get("status", {}).get("id") == 0:
            profiles[site] = data

    sites = list(profiles.keys())
    results = []

    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            site_a, site_b = sites[i], sites[j]
            result = score_same_person(
                profiles[site_a],
                profiles[site_b],
                llm,
                platform_a=site_a,
                platform_b=site_b,
            )
            if result.score >= min_score_to_report:
                results.append(result)

    return sorted(results, key=lambda r: -r.score)


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
