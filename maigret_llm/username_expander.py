"""
Module 2 — Expansão Inteligente de Usernames.

Gera variantes de username culturalmente informadas usando LLM,
muito além das permutações algorítmicas do --permute do Maigret.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .llm import BaseLLMClient

_SYSTEM = """\
Você é um especialista em OSINT com profundo conhecimento de convenções de username
em diferentes culturas, nichos e plataformas digitais.
Responda SOMENTE com JSON válido, sem texto adicional."""

_PROMPT_TEMPLATE = """\
## Dados do Alvo

- **Nome completo**: {full_name}
- **Nacionalidade/Cultura**: {nationality}
- **Ano de nascimento estimado**: {birth_year}
- **Contexto da investigação**: {context}
- **Usernames já conhecidos**: {known_usernames}
- **Informações adicionais**: {extra_info}

## Tarefa

Gere variantes de username que esta pessoa PROVAVELMENTE usa em outras plataformas.
Considere:
1. Padrões culturais regionais (ex: BR usa muitos números de nascimento, PT usa pontos)
2. Convenções por nicho (LinkedIn, GitHub, fóruns, gaming, adult content, crypto)
3. Variantes comuns do nome (diminutivos, apelidos, iniciais)
4. Padrões com números (anos, datas especiais)
5. Padrões com símbolos (ponto, underscore, hífen, sem separador)
6. Combinações com hobbies/profissão inferidos

## Formato de Resposta (JSON obrigatório)

```json
{{
  "variants": [
    {{
      "username": "johndoe90",
      "rationale": "padrão nome+ano de nascimento, muito comum em fóruns BR",
      "platforms": ["forums", "reddit", "twitter"],
      "confidence": 0.85
    }}
  ],
  "patterns_detected": ["uses birth year", "prefers lowercase", "no separators"],
  "cultural_notes": "Observações sobre padrões culturais detectados"
}}
```

Gere entre 15 e 30 variantes, ordenadas por probabilidade (maior primeiro).
Inclua variantes para diferentes nichos: profissional, social, gaming, técnico, crypto.
"""


@dataclass
class UsernameVariant:
    username: str
    rationale: str
    platforms: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class ExpansionResult:
    variants: list[UsernameVariant]
    patterns_detected: list[str]
    cultural_notes: str
    cost_usd: float = 0.0
    model: str = ""

    def usernames(self) -> list[str]:
        """Return sorted list of usernames (by confidence desc)."""
        return [v.username for v in sorted(self.variants, key=lambda x: -x.confidence)]


def expand_usernames(
    full_name: str,
    llm: BaseLLMClient,
    nationality: str = "brasileiro",
    birth_year: Optional[int] = None,
    context: str = "investigação OSINT geral",
    known_usernames: Optional[list[str]] = None,
    extra_info: str = "",
) -> ExpansionResult:
    """
    Gera variantes de username culturalmente informadas via LLM.

    Args:
        full_name: Nome completo da pessoa (ex: "João Pedro Almeida").
        llm: cliente LLM instanciado.
        nationality: Nacionalidade/cultura (ex: "brasileiro", "português", "americano").
        birth_year: Ano de nascimento estimado.
        context: Contexto da investigação (afeta os nichos priorizados).
        known_usernames: Lista de usernames já conhecidos.
        extra_info: Informações extras (hobbies, cidade, profissão, etc.).

    Returns:
        ExpansionResult com lista de variantes ranqueadas.
    """
    prompt = _PROMPT_TEMPLATE.format(
        full_name=full_name,
        nationality=nationality,
        birth_year=birth_year or "desconhecido",
        context=context,
        known_usernames=json.dumps(known_usernames or [], ensure_ascii=False),
        extra_info=extra_info or "nenhuma",
    )

    response = llm.complete(prompt, system=_SYSTEM, max_tokens=3000)
    parsed = _parse_response(response.content)

    variants = [
        UsernameVariant(
            username=v.get("username", ""),
            rationale=v.get("rationale", ""),
            platforms=v.get("platforms", []),
            confidence=float(v.get("confidence", 0.5)),
        )
        for v in parsed.get("variants", [])
        if v.get("username")
    ]

    return ExpansionResult(
        variants=variants,
        patterns_detected=parsed.get("patterns_detected", []),
        cultural_notes=parsed.get("cultural_notes", ""),
        cost_usd=response.cost_usd,
        model=response.model,
    )


def _parse_response(text: str) -> dict:
    """Extract JSON from LLM response (handles markdown code fences)."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract from ```json ... ``` block
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Find first { ... } block
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"variants": [], "patterns_detected": [], "cultural_notes": "parse error"}
