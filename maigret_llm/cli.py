"""
CLI — maigret-llm command line interface.

Usage:
    maigret-llm dossier <results.json> [options]
    maigret-llm expand <full_name> [options]
    maigret-llm maintain <data.json> [options]
    maigret-llm link <results.json> [options]
    maigret-llm parse <url> [options]
    maigret-llm prioritize <context> [options]
    maigret-llm agent <objective> [options]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .llm import create_llm


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maigret-llm",
        description="Maigret + LLM: OSINT inteligente com análise semântica",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  maigret-llm dossier results.json --provider anthropic
  maigret-llm expand "João Pedro Almeida" --nationality brasileiro --birth-year 1990
  maigret-llm agent "Investigar fraude de john_doe" --username john_doe --max-iter 8
  maigret-llm maintain /path/to/maigret/data.json --limit 50 --dry-run
  maigret-llm prioritize "fraude em criptomoedas, usuário ativo em fóruns"
        """,
    )

    # Global options
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "gemini", "ollama"],
        default=None,
        help="Provider LLM (padrão: auto-detecta por variável de ambiente)",
    )
    parser.add_argument("--model", default=None, help="Modelo específico do provider")
    parser.add_argument("--api-key", default=None, help="API key (ou use variáveis de ambiente)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- dossier ----
    p_dossier = subparsers.add_parser("dossier", help="Gera dossier narrativo a partir de results.json")
    p_dossier.add_argument("results", help="Arquivo JSON com resultados do Maigret")
    p_dossier.add_argument("--username", default="", help="Username investigado")
    p_dossier.add_argument(
        "--context",
        default="Analise padrões compatíveis com fraude financeira.",
        help="Contexto da investigação",
    )
    p_dossier.add_argument("--output", default=None, help="Arquivo de saída (.md)")

    # ---- expand ----
    p_expand = subparsers.add_parser("expand", help="Gera variantes de username via LLM")
    p_expand.add_argument("full_name", help="Nome completo do alvo")
    p_expand.add_argument("--nationality", default="brasileiro")
    p_expand.add_argument("--birth-year", type=int, default=None)
    p_expand.add_argument("--context", default="investigação OSINT geral")
    p_expand.add_argument("--known", nargs="*", default=[], help="Usernames já conhecidos")
    p_expand.add_argument("--extra-info", default="", help="Info extra (hobbies, cidade...)")

    # ---- maintain ----
    p_maintain = subparsers.add_parser(
        "maintain", help="Manutenção autônoma do data.json do Maigret"
    )
    p_maintain.add_argument("data_json", help="Caminho para o data.json do Maigret")
    p_maintain.add_argument("--maigret", default="maigret", help="Caminho do executável maigret")
    p_maintain.add_argument("--limit", type=int, default=20, help="Sites a verificar")
    p_maintain.add_argument(
        "--min-confidence", type=float, default=0.7, help="Confiança mínima para aplicar patch"
    )
    p_maintain.add_argument("--dry-run", action="store_true", help="Apenas mostra, não aplica")

    # ---- link ----
    p_link = subparsers.add_parser(
        "link", help="Detecta co-identidade entre perfis encontrados"
    )
    p_link.add_argument("results", help="Arquivo JSON com resultados do Maigret")
    p_link.add_argument("--min-score", type=int, default=50, help="Score mínimo para reportar")

    # ---- parse ----
    p_parse = subparsers.add_parser("parse", help="Parser adaptativo de página de perfil")
    p_parse.add_argument("url", help="URL do perfil")
    p_parse.add_argument("--platform", default="", help="Nome da plataforma")

    # ---- prioritize ----
    p_prio = subparsers.add_parser("prioritize", help="Prioriza sites para investigação")
    p_prio.add_argument("context", help="Contexto da investigação")
    p_prio.add_argument("--target-info", default="{}", help="JSON com info do alvo")
    p_prio.add_argument("--max-sites", type=int, default=100)

    # ---- agent ----
    p_agent = subparsers.add_parser("agent", help="Agente OSINT autônomo completo")
    p_agent.add_argument("objective", help="Objetivo da investigação")
    p_agent.add_argument("--username", default=None, help="Username inicial conhecido")
    p_agent.add_argument("--max-iter", type=int, default=10, help="Máximo de iterações")
    p_agent.add_argument("--maigret", default="maigret", help="Caminho do executável maigret")
    p_agent.add_argument("--output-dir", default=None, help="Diretório para salvar resultados")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    llm = create_llm(provider=args.provider, model=args.model, api_key=args.api_key)

    if args.command == "dossier":
        from .synthesis import synthesize_dossier
        results = json.loads(Path(args.results).read_text(encoding="utf-8"))
        result = synthesize_dossier(results, llm, username=args.username, fraud_context=args.context)
        output = result.markdown
        print(output)
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"\nSalvo em: {args.output}")
        print(f"\n[custo: ${result.cost_usd:.4f} | modelo: {result.model}]")

    elif args.command == "expand":
        from .username_expander import expand_usernames
        result = expand_usernames(
            full_name=args.full_name,
            llm=llm,
            nationality=args.nationality,
            birth_year=args.birth_year,
            context=args.context,
            known_usernames=args.known,
            extra_info=args.extra_info,
        )
        print(f"\nVariantes geradas ({len(result.variants)}):\n")
        for v in sorted(result.variants, key=lambda x: -x.confidence):
            print(f"  {v.username:<30} ({v.confidence:.0%})  — {v.rationale}")
        print(f"\nPadrões: {', '.join(result.patterns_detected)}")
        print(f"Notas culturais: {result.cultural_notes}")
        print(f"\n[custo: ${result.cost_usd:.4f}]")

    elif args.command == "maintain":
        from .site_maintenance import run_maintenance
        report = run_maintenance(
            data_json_path=Path(args.data_json),
            llm=llm,
            maigret_path=args.maigret,
            limit=args.limit,
            min_confidence=args.min_confidence,
            dry_run=args.dry_run,
        )
        print(f"\nSites verificados: {report.sites_checked}")
        print(f"Patches aplicados: {report.sites_patched}")
        for p in report.patches:
            status = "✓" if p.applied else ("⚠ baixa confiança" if not p.error else f"✗ {p.error}")
            print(f"  {p.site_name}: {status} | confiança {p.confidence:.0%}")
            if p.diagnosis:
                print(f"    → {p.diagnosis}")

    elif args.command == "link":
        from .identity_linker import link_all_profiles
        results = json.loads(Path(args.results).read_text(encoding="utf-8"))
        links = link_all_profiles(results, llm, min_score_to_report=args.min_score)
        if not links:
            print("Nenhuma correspondência de identidade encontrada.")
        for link in links:
            print(f"\n{link.platform_a} ↔ {link.platform_b}")
            print(f"  Score: {link.score}/100 — {link.verdict}")
            print(f"  Confiança: {link.confidence_label}")
            for e in link.evidence:
                print(f"  ✓ [{e.weight}] {e.detail}")
            for c in link.contradictions:
                print(f"  ✗ {c.detail}")

    elif args.command == "parse":
        from .adaptive_parser import parse_profile
        result = parse_profile(url=args.url, llm=llm, platform=args.platform)
        print(json.dumps(result.fields, ensure_ascii=False, indent=2))
        if result.parse_error:
            print(f"\nErro: {result.parse_error}", file=sys.stderr)
        print(f"\n[custo: ${result.cost_usd:.4f} | HTML: {result.raw_html_size:,} chars]")

    elif args.command == "prioritize":
        from .site_prioritizer import prioritize_sites
        try:
            target_info = json.loads(args.target_info)
        except json.JSONDecodeError:
            target_info = {}
        result = prioritize_sites(llm=llm, investigation_context=args.context, target_info=target_info)
        print(f"\nEstratégia: {result.reasoning}\n")
        for tier in result.tiers:
            print(f"[{tier.tier}]")
            print(f"  Sites: {', '.join(tier.sites)}")
            print(f"  Motivo: {tier.rationale}")
        top = result.ordered_sites[: args.max_sites]
        print(f"\nSites recomendados ({len(top)}): {', '.join(top[:20])}...")

    elif args.command == "agent":
        from .agent import OSINTAgent
        output_dir = Path(args.output_dir) if args.output_dir else None
        agent = OSINTAgent(
            llm=llm,
            maigret_path=args.maigret,
            max_iterations=args.max_iter,
            output_dir=output_dir,
        )
        result = agent.run(objective=args.objective, initial_username=args.username)
        print(f"\n{'='*60}")
        print(f"Investigação concluída!")
        print(f"Usernames investigados: {{', '.join(result.usernames_investigated)}")
        print(f"Perfis encontrados: {result.profiles_found}")
        print(f"Custo total: ${result.total_cost_usd:4f}")
        print(f"Tempo total: {result.total_duration_s:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
