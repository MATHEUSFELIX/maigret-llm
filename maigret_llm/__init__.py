"""
maigret_llm — Maigret + LLM: OSINT inteligente com análise semântica.

Módulos:
  1. synthesis        — Dossier narrativo a partir de resultados brutos
  2. username_expander — Expansão cultural de usernames
  3. site_maintenance — Manutenção autônoma do data.json
  4. identity_linker  — Detecção de co-identidade entre plataformas
  5. adaptive_parser  — Parser universal de páginas de perfil
  6. site_prioritizer — Priorização inteligente de sites
  7. agent            — Agente OSINT autônomo completo
"""

__version__ = "0.1.0"

from .llm import create_llm
from .synthesis import synthesize_dossier, DossierResult
from .username_expander import expand_usernames, ExpansionResult
from .site_maintenance import run_maintenance, analyze_broken_site, MaintenanceReport
from .identity_linker import score_same_person, link_all_profiles, LinkingResult
from .adaptive_parser import parse_profile, parse_profiles_from_maigret, ParsedProfile
from .site_prioritizer import prioritize_sites, PrioritizationResult
from .agent import OSINTAgent, OSINTAgentResult

__all__ = [
    "create_llm",
    "synthesize_dossier", "DossierResult",
    "expand_usernames", "ExpansionResult",
    "run_maintenance", "analyze_broken_site", "MaintenanceReport",
    "score_same_person", "link_all_profiles", "LinkingResult",
    "parse_profile", "parse_profiles_from_maigret", "ParsedProfile",
    "prioritize_sites", "PrioritizationResult",
    "OSINTAgent", "OSINTAgentResult",
]
