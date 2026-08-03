# GasMind AI

AI contract review for gas supply agreements.

This is the **MVP code scaffold**: a FastAPI + LangGraph pipeline that parses a
gas supply agreement (PDF/DOCX/TXT), inventories its clauses, scores risk,
detects missing clauses, and produces an evidence-backed executive summary with
negotiation recommendations.

> MVP scope is deliberately narrow. It is a **review assistant, not a draft or
> review tool**—it never invents clauses, legislation, or facts. Every
> conclusion cites the contract text (see `app/prompts/skills.py`).

## Repository layout

```
backend/
  app/
    main.py            # FastAPI entrypoint
    api/reviews.py     # POST /api/v1/reviews, GET /api/v1/health
    core/config.py     # settings from environment / .env
    llm/client.py      # LLM abstraction (mock | anthropic | openai | ollama)
    parsers/           # pdf / docx / txt normalization
    prompts/skills.py  # one prompt template per pipeline skill
    schemas/           # Pydantic output contracts (JSON schemas)
    workflow/          # LangGraph state, nodes, graph, service
    scripts/run_review.py  # CLI runner
  tests/
examples/
  contracts/           # sample gas supply agreement
  output/              # generated review JSON
eval/
  test_contracts/      # labelled corpus for measuring precision/recall
docs/
```

## Running

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# CLI (uses the deterministic mock LLM — no API key needed)
python -m app.scripts.run_review ../examples/contracts/sample_gas_supply_agreement.txt

# API
uvicorn app.main:app --reload
# open http://127.0.0.1:8000/docs
```

## Configuring a real LLM

Copy `.env.example` to `.env` and set `LLM_PROVIDER`.

- `anthropic` (default model `claude-sonnet-4-20250514`)
- `openai`
- `ollama`

The pipeline is provider-agnostic; the mock is deterministic and offline so the
test suite and CLI work with no API key.

## Tests

```bash
cd backend
pytest -q
```

## Design notes

- **Traceable evidence**: every risk conclusion carries `source_spans` pointing
  back into the contract text (PRD AI principles).
- **JSON contracts**: all output models live in `app/schemas/` and are shared
  by the pipeline and the API.
- **LanguageGraph**: the workflow is a compiled graph in
  `app/workflow/graph.py`; nodes are pure functions in `nodes.py`, individually
  testable and replaceable.

## Future work (non-goals for v1)

Contract drafting, Word redlining, multi-user collaboration, version
comparison, e-signatures, billing, workflow automation, case-law retrieval.
See PRD for the full roadmap.