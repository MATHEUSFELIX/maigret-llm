"""
Module 3 — Manutenção Autônoma do data.json.

Pipeline: --self-check detecta site quebrado → LLM analisa HTML atual →
LLM propõe patch no data.json → validação → aplicação.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from .llm import BaseLLMClient

_SYSTEM = """\
Você é um engenheiro especializado em web scraping e manutenção de bases de dados OSINT.
Sua tarefa é analisar HTML de páginas de perfil e propor patches no formato data.json do Maigret.
Responda SOMENTE com JSON válido, sem texto adicional."""

_ANALYSIS_PROMPT = """\
## Site com Problema: {site_name}

### Configuração Atual no data.json
```json
{current_config}
```

### Erro Detectado pelo --self-check
```
{error_message}
```

### HTML da Página de Perfil (amostra)
```html
{html_sample}
```

### HTML da Página "Usuário Não Encontrado" (amostra)
```html
{not_found_html}
```

## Tarefa

Analise o HTML e a configuração atual. Identifique o problema e proponha um patch.

Os campos mais comuns que precisam atualização:
- `errorType`: como o site indica usuário não encontrado ("status_code", "message", "response_url", "redirect_url")
- `errorMsg`: texto/regex que aparece quando usuário não existe
- `urlMain`: URL principal do site
- `url`: padrão de URL do perfil
- `tags`: categorias do site

Responda com JSON no formato:
```json
{{
  "diagnosis": "Explicação do problema encontrado",
  "confidence": 0.85,
  "patch": {{
    "errorType": "message",
    "errorMsg": "User not found"
  }},
  "full_updated_config": {{
    // configuração completa atualizada (copie e modifique a atual)
  }},
  "test_recommendation": "Como testar se o patch funciona"
}}
```
"""


@dataclass
class SitePatch:
    site_name: str
    diagnosis: str
    confidence: float
    patch: dict[str, Any]
    full_updated_config: dict[str, Any]
    test_recommendation: str
    applied: bool = False
    error: str = ""


@dataclass
class MaintenanceReport:
    patches: list[SitePatch]
    sites_checked: int
    sites_patched: int
    cost_usd: float = 0.0


def run_self_check(maigret_path: str = "maigret", limit: int = 20) -> list[dict]:
    """
    Roda o --self-check do Maigret e retorna lista de sites com erro.

    Args:
        maigret_path: Caminho para o executável maigret.
        limit: Número máximo de sites a verificar.
    """
    try:
        result = subprocess.run(
            [maigret_path, "--self-check", "--json", f"--top-sites={limit}"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout
        # Try to parse JSON output
        try:
            data = json.loads(output)
            if isinstance(data, dict):
                return [
                    {"site": k, **v}
                    for k, v in data.items()
                    if v.get("status") not in ("ok", "good")
                ]
        except json.JSONDecodeError:
            pass

        # Fallback: parse text output
        broken = []
        for line in output.splitlines():
            if "ERROR" in line or "BROKEN" in line or "False positive" in line:
                parts = line.split()
                if parts:
                    broken.append({"site": parts[0], "error": line.strip()})
        return broken
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[site_maintenance] Aviso: não foi possível rodar maigret: {e}")
        return []


def fetch_html(url: str, timeout: int = 15) -> str:
    """Fetch page HTML with a browser-like user agent."""
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except URLError as e:
        return f"<!-- fetch error: {e} -->"


def analyze_broken_site(
    site_name: str,
    current_config: dict[str, Any],
    llm: BaseLLMClient,
    error_message: str = "",
    test_username: str = "testuser123notexist",
) -> SitePatch:
    """
    Analisa um site quebrado e propõe um patch via LLM.

    Args:
        site_name: Nome do site no data.json.
        current_config: Configuração atual do site.
        llm: cliente LLM.
        error_message: Mensagem de erro do --self-check.
        test_username: Username sabidamente inexistente para testar.
    """
    url_template = current_config.get("url", "")
    profile_url = url_template.replace("{}", test_username) if "{}" in url_template else ""
    not_found_url = url_template.replace("{}", "zzz_this_user_does_not_exist_xyz_999")

    html_sample = ""
    not_found_html = ""

    if profile_url:
        html_sample = fetch_html(profile_url)[:8000]
    if not_found_url and not_found_url != profile_url:
        not_found_html = fetch_html(not_found_url)[:8000]

    prompt = _ANALYSIS_PROMPT.format(
        site_name=site_name,
        current_config=json.dumps(current_config, ensure_ascii=False, indent=2),
        error_message=error_message or "Erro não especificado (detectado pelo self-check)",
        html_sample=html_sample or "(não foi possível buscar o HTML)",
        not_found_html=not_found_html or "(não foi possível buscar o HTML de 'não encontrado')",
    )

    response = llm.complete(prompt, system=_SYSTEM, max_tokens=3000)
    parsed = _parse_json(response.content)

    return SitePatch(
        site_name=site_name,
        diagnosis=parsed.get("diagnosis", ""),
        confidence=float(parsed.get("confidence", 0.0)),
        patch=parsed.get("patch", {}),
        full_updated_config=parsed.get("full_updated_config", {}),
        test_recommendation=parsed.get("test_recommendation", ""),
    )


def apply_patch(
    data_json_path: Path,
    patch: SitePatch,
    min_confidence: float = 0.7,
    dry_run: bool = False,
) -> bool:
    """
    Aplica o patch no data.json se a confiança for suficiente.

    Args:
        data_json_path: Caminho para o data.json do Maigret.
        patch: SitePatch gerado pelo analyze_broken_site.
        min_confidence: Confiança mínima para aplicar (0-1).
        dry_run: Se True, apenas mostra o que seria feito sem aplicar.

    Returns:
        True se aplicado com sucesso.
    """
    if patch.confidence < min_confidence:
        print(
            f"[{patch.site_name}] Confiança {patch.confidence:.0%} abaixo do mínimo "
            f"{min_confidence:.0%} — patch não aplicado."
        )
        return False

    if dry_run:
        print(f"[DRY RUN] {patch.site_name}: aplicaria patch com confiança {patch.confidence:.0%}")
        print(json.dumps(patch.patch, indent=2, ensure_ascii=False))
        return True

    try:
        data = json.loads(data_json_path.read_text(encoding="utf-8"))
        if patch.site_name in data:
            if patch.full_updated_config:
                data[patch.site_name] = patch.full_updated_config
            else:
                data[patch.site_name].update(patch.patch)
            data_json_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            patch.applied = True
            print(f"[{patch.site_name}] Patch aplicado com confiança {patch.confidence:.0%}.")
            return True
        else:
            patch.error = f"Site {patch.site_name!r} não encontrado no data.json"
            return False
    except Exception as e:
        patch.error = str(e)
        return False


def run_maintenance(
    data_json_path: Path,
    llm: BaseLLMClient,
    maigret_path: str = "maigret",
    limit: int = 20,
    min_confidence: float = 0.7,
    dry_run: bool = False,
) -> MaintenanceReport:
    """
    Pipeline completo de manutenção autônoma.

    1. Roda --self-check
    2. Para cada site quebrado, busca HTML e pede análise ao LLM
    3. Aplica patches com confiança suficiente
    """
    broken_sites = run_self_check(maigret_path, limit)
    print(f"Sites quebrados detectados: {len(broken_sites)}")

    data = json.loads(data_json_path.read_text(encoding="utf-8"))
    patches: list[SitePatch] = []
    total_cost = 0.0

    for entry in broken_sites:
        site_name = entry.get("site", "")
        if not site_name or site_name not in data:
            continue

        print(f"\nAnalisando: {site_name}...")
        patch = analyze_broken_site(
            site_name=site_name,
            current_config=data[site_name],
            llm=llm,
            error_message=entry.get("error", ""),
        )
        total_cost += 0.0  # accumulated per response if needed

        apply_patch(data_json_path, patch, min_confidence=min_confidence, dry_run=dry_run)
        patches.append(patch)

    applied = sum(1 for p in patches if p.applied)
    return MaintenanceReport(
        patches=patches,
        sites_checked=len(broken_sites),
        sites_patched=applied,
        cost_usd=total_cost,
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
