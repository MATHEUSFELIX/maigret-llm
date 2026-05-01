# maigret-llm

**Maigret + LLM**: OSINT inteligente com análise semântica de identidade digital.

Integra o [Maigret](https://github.com/soxoj/maigret) com modelos de linguagem (Anthropic Claude, OpenAI GPT, Google Gemini, Ollama) para enriquecer investigações OSINT com síntese narrativa, expansão de usernames, detecção de co-identidade, parser adaptativo de perfis, priorização de sites e um agente autônomo completo.

---

## Funcionalidades

| Módulo | Descrição |
|---|---|
| `synthesis` | Gera dossiê narrativo em Markdown a partir de resultados do Maigret |
| `username_expander` | Gera variantes culturais de username via LLM |
| `site_maintenance` | Manutenção autônoma do `data.json` do Maigret |
| `identity_linker` | Detecta co-identidade entre perfis encontrados |
| `adaptive_parser` | Parser adaptativo de páginas de perfil via LLM |
| `site_prioritizer` | Prioriza sites para investigação por contexto |
| `agent` | Agente OSINT autônomo (loop decisão→ação→observação) |

---

## Instalação

```bash
# Apenas o core
pip install maigret-llm

# Com provedor específico
pip install "maigret-llm[anthropic]"
pip install "maigret-llm[openai]"
pip install "maigret-llm[gemini]"

# Todos os provedores
pip install "maigret-llm[all]"
```

O Maigret precisa estar instalado separadamente:

```bash
pip install maigret
```

---

## Configuração

O provedor LLM é detectado automaticamente pela variável de ambiente:

| Variável | Provedor |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic Claude |
| `OPENAI_API_KEY` | OpenAI GPT |
| `GEMINI_API_KEY` | Google Gemini |
| *(nenhuma)* | Ollama (local) |

---

## Uso via CLI

```bash
# Gerar dossiê narrativo
maigret-llm dossier results.json --username joao_silva --provider anthropic

# Expandir variantes de username
maigret-llm expand "João Pedro Almeida" --nationality brasileiro --birth-year 1990

# Agente OSINT autônomo completo
maigret-llm agent "Investigar fraude de john_doe" --username john_doe --max-iter 8

# Manutenção do data.json do Maigret
maigret-llm maintain /path/to/maigret/data.json --limit 50 --dry-run

# Detectar co-identidade entre perfis
maigret-llm link results.json --min-score 60

# Parser adaptativo de perfil
maigret-llm parse https://reddit.com/user/exemplo

# Priorizar sites para investigação
maigret-llm prioritize "fraude em criptomoedas, usuário ativo em fóruns"
```

---

## Uso via Python

```python
import json
from maigret_llm.llm import create_llm
from maigret_llm.synthesis import synthesize_dossier
from maigret_llm.username_expander import expand_usernames
from maigret_llm.agent import OSINTAgent

# Criar cliente LLM (auto-detecta pelo ambiente)
llm = create_llm()

# Gerar dossiê a partir de resultados do Maigret
with open("results.json") as f:
    results = json.load(f)

dossier = synthesize_dossier(
    maigret_results=results,
    llm=llm,
    username="target_user",
    fraud_context="Investigar possível fraude financeira.",
)
print(dossier.markdown)

# Expandir usernames
expansion = expand_usernames(
    full_name="João Pedro Almeida",
    llm=llm,
    nationality="brasileiro",
    birth_year=1990,
)
for v in expansion.variants:
    print(f"{v.username} ({v.confidence:.0%}) — {v.rationale}")

# Agente autônomo
agent = OSINTAgent(llm=llm, max_iterations=10)
result = agent.run(
    objective="Investigar fraude de john_doe",
    initial_username="john_doe",
)
print(result.to_markdown())
```

---

## Provedores LLM

```python
from maigret_llm.llm import create_llm

# Auto-detect (recomendado)
llm = create_llm()

# Explícito
llm = create_llm(provider="anthropic", model="claude-opus-4-5")
llm = create_llm(provider="openai", model="gpt-4o")
llm = create_llm(provider="gemini", model="gemini-1.5-pro")
llm = create_llm(provider="ollama", model="llama3")
```

---

## Estrutura do Projeto

```
maigret_llm/
├── __init__.py
├── llm/
│   ├── __init__.py       # Factory: create_llm()
│   ├── base.py           # BaseLLMClient
│   ├── anthropic.py      # AnthropicClient
│   ├── openai.py         # OpenAIClient
│   ├── gemini.py         # GeminiClient
│   └── ollama.py         # OllamaClient
├── synthesis.py          # Módulo 1: Dossiê narrativo
├── username_expander.py  # Módulo 2: Expansão de usernames
├── site_maintenance.py   # Módulo 3: Manutenção de data.json
├── identity_linker.py    # Módulo 4: Co-identidade
├── adaptive_parser.py    # Módulo 5: Parser adaptativo
├── site_prioritizer.py   # Módulo 6: Priorização de sites
├── agent.py              # Módulo 7: Agente OSINT autônomo
└── cli.py                # Interface de linha de comando
```

---

## Licença

MIT
