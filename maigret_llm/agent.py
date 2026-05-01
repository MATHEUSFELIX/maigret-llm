"""
Module 7 — Agente OSINT Completo.

O Maigret vira uma tool dentro de um loop de agente LLM.
O agente decide quando buscar, quais usernames tentar, quando cruzar dados,
e quando gerar o dossier final — sem intervenção manual.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .llm import BaseLLMClient
from .synthesis import synthesize_dossier, DossierResult
from .username_expander import expand_usernames, ExpansionResult
from .identity_linker import score_same_person, LinkingResult
from .site_prioritizer import prioritize_sites, PrioritizationResult
from .adaptive_parser import parse_profile

_SYSTEM = """\
Você é um agente OSINT autônomo especializado em investigação de identidade digital.
Você tem acesso a ferramentas de busca e análise. Sua missão é investigar o alvo
da forma mais completa possível, tomando decisões táticas ao longo do processo.

Ao receber o objetivo de investigação, você deve:
1. Planejar quais usernames investigar e em quais plataformas
2. Interpretar os resultados e decidir os próximos passos
3. Cruzar dados entre plataformas
4. Gerar um dossier final completo

Responda SEMPRE com JSON válido descrevendo sua próxima ação.
"""

_DECISION_PROMPT = """\
## Estado Atual da Investigação

**Objetivo**: {objective}

**Iteração**: {iteration}/{max_iterations}

**Usernames já investigados**: {investigated_usernames}

**Resultados acumulados** (resumo):
```json
{accumulated_summary}
```

**Histórico de decisões**:
{decision_history}

## Ferramentas Disponíveis

- `search_username`: Busca um username no Maigret
- `expand_usernames`: Gera variantes de username via LLM
- `parse_profile`: Extrai dados detalhados de uma URL de perfil
- `link_profiles`: Analisa co-identidade entre dois perfis
- `generate_dossier`: Gera o dossier final e encerra a investigação
- `stop`: Encerra sem dossier (quando não há dados suficientes)

## Próxima Ação

Com base no estado atual, qual é a ação mais valiosa agora?

Responda com JSON:
```json
{{
  "reasoning": "Por que esta ação agora",
  "action": "search_username",
  "params": {{
    "username": "john_doe",
    "context": "username alternativo encontrado no Reddit"
  }}
}}
```

Ações possíveis e seus parâmetros:
- `search_username`: {{"username": "...", "sites": ["GitHub", "Reddit"]}}
- `expand_usernames`: {{"full_name": "...", "known_usernames": [...], "context": "..."}}
- `parse_profile`: {{"url": "...", "platform": "..."}}
- `link_profiles`: {{"site_a": "...", "site_b": "..."}}
- `generate_dossier`: {{"focus": "fraude | identidade | geral"}}
- `stop`: {{"reason": "..."}}
"""


@dataclass
class AgentAction:
    reasoning: str
    action: str
    params: dict[str, Any]


@dataclass
class AgentStep:
    iteration: int
    action: AgentAction
    result: Any
    duration_s: float = 0.0
    cost_usd: float = 0.0


@dataclass
class OSINTAgentResult:
    objective: str
    steps: list[AgentStep]
    dossier: Optional[DossierResult]
    total_cost_usd: float
    total_duration_s: float
    usernames_investigated: list[str]
    profiles_found: int

    def to_markdown(self) -> str:
        lines = [
            f"# Relatório OSINT: {self.objective}",
            "",
            f"**Usernames investigados**: {{', '.join(self.usernames_investigated)}",
            f"**Perfis encontrados**: {self.profiles_found}",
            f"**Custo total**: ${self.total_cost_usd:.4f}",
            f"**Tempo total**: {self.total_duration_s:.1f}s",
            f"**Iterações**: {len(self.steps)}",
            "",
            "---",
            "",
        ]
        if self.dossier:
            lines.append(self.dossier.markdown)
        return "\n".join(lines)


class OSINTAgent:
    """
    Agente OSINT autônomo └ loop de decisão-ação-observação.

    O agente usa o LLM para decidir cada próximo passo com base
    nos resultados acumulados, sem intervenção manual.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        maigret_path: str = "maigret",
        max_iterations: int = 10,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.llm = llm
        self.maigret_path = maigret_path
        self.max_iterations = max_iterations
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "maigret_llm_agent"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # State
        self._results: dict[str, Any] = {}  # username -> maigret results
        self._steps: list[AgentStep] = []
        self._investigated: list[str] = []
        self._decisions: list[str] = []
        self._total_cost = 0.0

    def run(self, objective: str, initial_username: Optional[str] = None) -> OSINTAgentResult:
        """
        Executa a investigação completa de forma autônoma.

        Args:
            objective: Descrição do objetivo (ex: "Investigar fraude de João Almeida").
            initial_username: Username inicial se conhecido.

        Returns:
            OSINTAgentResult com dossier completo.
        """
        start_time = time.time()
        self._log(f"\n{'='*60}")
        self._log(f"OSINT AGENT INICIADO")
        self._log(f"Objetivo: {objective}")
        self._log(f"{'='*60}\n")

        # If we have an initial username, search it first
        if initial_username:
            self._log(f"Busca inicial: {initial_username}")
            self._do_search_username(initial_username)
            self._investigated.append(initial_username)

        dossier: Optional[DossierResult] = None

        for iteration in range(1, self.max_iterations + 1):
            self._log(f"\n--- Iteração {iteration}/{self.max_iterations} ---")

            step_start = time.time()
            action = self._decide_next_action(objective, iteration)

            self._log(f"Ação: {action.action}")
            self._log(f"Raciocínio: {action.reasoning}")

            result = None

            if action.action == "search_username":
                username = action.params.get("username", "")
                if username and username not in self._investigated:
                    result = self._do_search_username(
                        username,
                        sites=action.params.get("sites"),
                    )
                    self._investigated.append(username)
                    self._decisions.append(
                        f"[{iteration}] Buscou username '{username}' — {action.reasoning}"
                    )

            elif action.action == "expand_usernames":
                result = self._do_expand_usernames(
                    full_name=action.params.get("full_name", ""),
                     known_usernames=action.params.get("known_usernames", self._investigated),
                    context=action.params.get("context", objective),
                )
                # Auto-search the top 3 new suggestions
                if result:
                    for variant in result.variants[:3]:
                        if variant.username not in self._investigated:
                            self._do_search_username(variant.username)
                            self._investigated.append(variant.username)
                self._decisions.append(f"[{iteration}] Expandiu usernames — {action.reasoning}")

            elif action.action == "parse_profile":
                url = action.params.get("url", "")
                if url:
                    result = parse_profile(url, self.llm, platform=action.params.get("platform", ""))
                    self._total_cost += result.cost_usd
                self._decisions.append(f"[{iteration}] Parseou perfil em {url[:50]}")

            elif action.action == "link_profiles":
                site_a = action.params.get("site_a", "")
                site_b = action.params.get("site_b", "")
                result = self._do_link_profiles(site_a, site_b)
                if result:
                    self._decisions.append(
                        f"[{iteration}] Linkagem {site_a}↔{site_b}: score={result.score}"
                    )

            elif action.action == "generate_dossier":
                self._log("Gerando dossier final...")
                dossier = self._do_generate_dossier(objective)
                step = AgentStep(
                    iteration=iteration,
                    action=action,
                    result=dossier,
                    duration_s=time.time() - step_start,
                )
                self._steps.append(step)
                break

            elif action.action == "stop":
                self._log(f"Agente encerrou: {action.params.get('reason', '')}")
                break

            step = AgentStep(
                iteration=iteration,
                action=action,
                result=result,
                duration_s=time.time() - step_start,
            )
            self._steps.append(step)

        total_duration = time.time() - start_time
        profiles_found = sum(
            sum(1 for v in r.values() if isinstance(v, dict) and v.get("status", {}).get("id") == 0)
            for r in self._results.values()
            if isinstance(r, dict)
        )

        final = OSINTAgentResult(
            objective=objective,
            steps=self._steps,
            dossier=dossier,
            total_cost_usd=self._total_cost,
            total_duration_s=total_duration,
            usernames_investigated=self._investigated,
            profiles_found=profiles_found,
        )

        # Save report
        report_path = self.output_dir / "agent_report.md"
        report_path.write_text(final.to_markdown(), encoding="utf-8")
        self._log(f"\nRelatório salvo em: {report_path}")

        return final

    # ------------------------------------------------------------------ #
    # Private action methods                                               #         
    # ------------------------------------------------------------------ #

    def _decide_next_action(self, objective: str, iteration: int) -> AgentAction:
        """Ask the LLM what to do next."""
        accumulated_summary = self._build_accumulated_summary()

        prompt = _DECISION_PROMPT.format(
            objective=objective,
            iteration=iteration,
            max_iterations=self.max_iterations,
            investigated_usernames=json.dumps(self._investigated, ensure_ascii=False),
            accumulated_summary=json.dumps(accumulated_summary, ensure_ascii=False, indent=2)[:4000],
            decision_history="\n".join(self._decisions[-5:]) or "(nenhuma ainda)",
        )

        response = self.llm.complete(prompt, system=_SYSTEM, max_tokens=1000)
        self._total_cost += response.cost_usd

        parsed = _parse_json(response.content)
        return AgentAction(
            reasoning=parsed.get("reasoning", ""),
            action=parsed.get("action", "stop"),
            params=parsed.get("params", {}),
        )

    def _do_search_username(
        self, username: str, sites: Optional[list[str]] = None
    ) -> Optional[dict]:
        """Run Maigret for a username and store results."""
        self._log(f"  Maigret: buscando '{username}'...")
        out_file = self.output_dir / f"{username}.json"

        cmd = [
            self.maigret_path,
            username,
            "--json", str(out_file),
            "--no-color",
        ]
        if sites:
            cmd += ["--site", ",".join(sites)]

        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
            if out_file.exists():
                results = json.loads(out_file.read_text(encoding="utf-8"))
                self._results[username] = results
                found = sum(
                    1 for v in results.values()
                    if isinstance(v, dict) and v.get("status", {}).get("id") == 0
                )
                self._log(f"  → {found} perfis encontrados para '{username}'")
                return results
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            self._log(f"  → Erro ao buscar '{username}': {e}")
        return None

    def _do_expand_usernames(
        self, full_name: str, known_usernames: list[str], context: str
    ) -> Optional[ExpansionResult]:
        """Expand usernames via LLM."""
        if not full_name:
            return None
        result = expand_usernames(
            full_name=full_name,
            llm=self.llm,
            known_usernames=known_usernames,
            context=context,
        )
        self._total_cost += result.cost_usd
        self._log(f"  → {len(result.variants)} variantes geradas")
        return result

    def _do_link_profiles(self, site_a: str, site_b: str) -> Optional[LinkingResult]:
        """Compare two profiles for co-identity."""
        # Find profiles across all investigated usernames
        profile_a = self._find_profile(site_a)
        profile_b = self._find_profile(site_b)
        if not profile_a or not profile_b:
            return None
        result = score_same_person(profile_a, profile_b, self.llm, site_a, site_b)
        self._total_cost += result.cost_usd
        self._log(f"  → Co-identidade {site_a}↔{site_b}: score={result.score} ({result.verdict})")
        return result

    def _do_generate_dossier(self, objective: str) -> DossierResult:
        """Merge all results and generate the final dossier."""
        merged: dict[str, Any] = {}
        for username, results in self._results.items():
            if isinstance(results, dict):
                for site, data in results.items():
                    if site not in merged:
                        merged[site] = data

        result = synthesize_dossier(
            maigret_results=merged,
            llm=self.llm,
            username=" / ".join(self._investigated),
            fraud_context=f"Contexto da investigação: {objective}",
        )
        self._total_cost += result.cost_usd
        return result

    def _find_profile(self, site_name: str) -> Optional[dict]:
        """Find a site's data across all stored results."""
        for results in self._results.values():
            if isinstance(results, dict) and site_name in results:
                return results[site_name]
        return None

    def _build_accumulated_summary(self) -> dict:
        """Build a condensed summary of all findings so far."""
        summary: dict[str, Any] = {
            "usernames_investigated": self._investigated,
            "total_profiles_found": 0,
            "sites_with_findings": [],
        }
        for username, results in self._results.items():
            if not isinstance(results, dict):
                continue
            for site, data in results.items():
                if isinstance(data, dict) and data.get("status", {}).get("id") == 0:
                    summary["total_profiles_found"] += 1  # type: ignore[operator]
                    entry = {"username": username, "site": site}
                    if data.get("url_user"):
                        entry["url"] = data["url_user"]
                    if data.get("ids_data"):
                        entry["ids"] = data["ids_data"]
                    summary["sites_with_findings"].append(entry)  # type: ignore[union-attr]
        return summary

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


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
    match = re.search(r'\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}
