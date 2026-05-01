"""
Module 1 — Interpretação e Síntese dos Resultados.

Recebe o JSON bruto do Maigret e produz um dossier narrativo estruturado.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .llm import BaseLLMClient

_SYSTEM = """\
Você é um analista OSINT sênior especializado em investigação de identidade digital.
Sua análise deve ser objetiva, baseada em evidências e livre de especulação não fundamentada.
Sempre cite quais dados sustentam cada conclusão.
Responda SEMPRE em português brasileiro, formato Markdown."""

_PROMPT_TEMPLATE = """\
## Dados Brutos do Maigret

```json
{results_json}
```

## Sua Tarefa

Com base nos dados acima, produza um **Dossier de Identidade Digital** estruturado contendo:

### 1. Resumo Executivo
Síntese em 3-5 linhas do que foi encontrado e qual o nível de exposição digital do alvo.

### 2. Perfil de Plataformas
- Quais categorias de plataformas o usuário frequenta (profissional, social, técnica, adulta, gaming, etc.)
- Padrão temporal: quando criou contas, frequência estimada de atividade
- Plataformas de maior Valor investigativo e por quê

### 3. Inconsistências e Alertas
- Contradições entre plataformas (ex: perfil conservador + plataforma adulta)
- Usernames alternativos encontrados
- Dados que divergem entre plataformas

### 4. Perfil Inferido
- Faixa etária estimada (com base em evidências)
- Localização provável (com base em evidências)
- Área de atuação profissional inferida
- Interesses e hobbies detectados

### 5. Dados de Alto Valor para Investigação
Lista dos identificadores mais relevantes encontrados (emails, IDs, links, etc.)

### 6. Indicadores de Risco / Fraude
{fraud_context}

### 7. Próximos Passos Recomendados
Quais fontes adicionais investigar e por quê.

---
Baseie CADA afirmação em dados concretos encontrados. Se não houver evidência, diga explicitamente.
"""


@dataclass
class DossierResult:
    markdown: str
    username: str
    sites_found: int
    sites_checked: int
    cost_usd: float = 0.0
    model: str = ""


def synthesize_dossier(
    maigret_results: dict[str, Any],
    llm: BaseLLMClient,
    username: str = "",
    fraud_context: str = "Analise se há padrões compatíveis com fraude financeira, phishing ou engenharia social.",
) -> DossierResult:
    """
    Gera um dossier narrativo a partir dos resultados brutos do Maigret.

    Args:
        maigret_results: dict retornado pelo Maigret (ou carregado de JSON).
        llm: cliente LLM instanciado.
        username: username investigado (para o título do dossier).
        fraud_context: instrução adicional sobre o tipo de investigação.

    Returns:
        DossierResult com o markdown do dossier e metadados.
    """
    # Simplify the results to avoid sending too many tokens
    simplified = _simplify_results(maigret_results)
    results_json = json.dumps(simplified, ensure_ascii=False, indent=2)

    # Trim if too large (keep under ~60k chars)
    if len(results_json) > 60_000:
        results_json = results_json[:60_000] + "\n... [truncado para caber no contexto]"

    prompt = _PROMPT_TEMPLATE.format(
        results_json=results_json,
        fraud_context=fraud_context,
    )

    response = llm.complete(prompt, system=_SYSTEM)

    sites_found = sum(
        1 for v in maigret_results.values()
        if isinstance(v, dict) and v.get("status", {}).get("id") == 0
    )
    sites_checked = len(maigret_results)

    return DossierResult(
        markdown=response.content,
        username=username,
        sites_found=sites_found,
        sites_checked=sites_checked,
        cost_usd=response.cost_usd,
        model=response.model,
    )


def _simplify_results(results: dict[str, Any]) -> dict[str, Any]:
    """Keep only relevant fields per site to reduce token usage."""
    simplified: dict[str, Any] = {}
    for site, data in results.items():
        if not isinstance(data, dict):
            continue
        status = data.get("status", {})
        if isinstance(status, dict):
            status_id = status.get("id")
        else:
            status_id = None

        entry: dict[str, Any] = {
            "found": status_id == 0,
            "url": data.get("url_user") or data.get("url"),
        }

        # Include extracted IDs if present
        ids = data.get("ids_data") or data.get("ids") or {}
        if ids:
            entry["ids"] = ids

        # Include any profile fields extracted
        for key in ("name", "bio", "location", "created_at", "followers", "following"):
            if data.get(key):
                entry[key] = data[key]

        if entry["found"] or ids:
            simplified[site] = entry

    return simplified
