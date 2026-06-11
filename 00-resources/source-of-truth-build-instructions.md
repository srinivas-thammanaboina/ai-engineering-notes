# Source of Truth Generation — Build Instructions

Instructions for generating a two-layer knowledge corpus from project repos, intended to be executed with an AI coding agent (Copilot Agent mode / Claude) working repo-by-repo. The output feeds a RAG + lineage-graph system that answers cross-repo questions about merchant data pipelines.

---

## 0. Target Architecture (what we are producing)

```
knowledge-base/                          <- new dedicated git repo
├── registry/
│   └── fields/                          <- Layer 0: canonical field registry (YAML, human-reviewed)
│       └── merchant.category_code.yaml
├── extracted/                           <- Layer 1: machine-extracted structured facts (JSON, regenerated)
│   ├── transformations/
│   ├── validations/
│   ├── flow_edges/
│   └── coverage_report.json
├── docs/                                <- Layer 2: rendered knowledge documents (Markdown, embedded for RAG)
│   ├── fields/                          <- one "field card" per field_id
│   ├── jobs/                            <- one doc per Spark job / scheduled query
│   ├── datasets/                        <- one doc per table / keyspace / column family
│   ├── apis/                            <- one doc per endpoint
│   └── authored/                        <- human-written: architecture, decisions, runbooks
└── graph/
    └── lineage.json                     <- nodes + edges, queryable by the agent as a tool
```

**Rules that apply everywhere:**
- Every artifact references fields ONLY by `field_id` (e.g. `merchant.category_code`), never by raw column name. Raw names live in the registry as aliases.
- Every generated file carries provenance: `source_repo`, `source_commit`, `source_file`, `generated_at`.
- Layer 1 and Layer 2 are fully regenerable. Never hand-edit them; fix the extractor or the source instead. Human knowledge goes ONLY in `registry/` and `docs/authored/`.

---

## 1. Phase 1 — Canonical Field Registry (Layer 0)

**Do this first. Everything else keys off it.**

### 1.1 Inventory schemas
For each repo / system, have the agent enumerate every schema it can find:
- Hive DDLs, BigQuery table definitions / Terraform schemas
- Spark job input/output schemas (StructType definitions, case classes)
- Bigtable column families + qualifiers, Couchbase document models
- API DTOs / OpenAPI specs
- UI form models / shared validation schemas

Output one CSV per system: `system, container (table/endpoint/component), field_raw_name, type, nullable, sample_description`.

### 1.2 Cluster aliases into canonical fields
Prompt the agent to fuzzy-group raw names that represent the same logical field across systems (`mer_cat_cd` ≈ `merchantCategoryCode` ≈ `mcc`). **Output is a PROPOSAL — a human must review every grouping before it is committed.** Wrong alias grouping silently corrupts every downstream answer.

### 1.3 Registry record format (one YAML file per field)

```yaml
field_id: merchant.category_code        # namespace.snake_case, immutable once assigned
canonical_name: merchant_category_code
description: >
  ISO 18245 merchant category code assigned during enrichment.
business_meaning: >                     # human-authored: why it exists, who consumes it
data_type: string
pii_classification: none                # none | internal | confidential | restricted
owner: team-merchant-enrichment
aliases:
  - {system: hive,      container: stg_merchant_raw,        name: mer_cat_cd}
  - {system: bigquery,  container: enriched.merchant_dim,   name: merchant_category_code}
  - {system: bigtable,  container: merchant_profile/attrs,  name: mcc}
  - {system: api,       container: GET /v2/merchants/{id},  name: $.profile.mcc}
  - {system: ui,        container: MerchantProfileForm,     name: mcc}
status: active                          # active | deprecated
notes: []                               # gotchas, history, tribal knowledge
```

**Acceptance gate:** registry covers ≥ the top N business-critical fields before moving on (start with ~20–50, not all). Each file PR-reviewed by the owning team.

---

## 2. Phase 2 — Extraction (Layer 1)

Run per-repo. Three record types, common JSON schema. The agent should emit records it is CONFIDENT about and a separate `_uncertain.json` list for anything ambiguous (dynamic SQL, reflection, config-driven mappings) — uncertain items get human annotation, never guessed.

### 2.1 Transformation records

```json
{
  "id": "txf::enrich-merchant-job::mcc_derivation",
  "type": "transformation",
  "inputs": ["merchant.raw_sic_code", "merchant.industry_text"],
  "output": "merchant.category_code",
  "logic_summary": "Maps SIC to MCC via ref table mcc_sic_xref; falls back to NLP classifier output when SIC null.",
  "logic_kind": "lookup+fallback",
  "job": "enrich-merchant-job",
  "layer": "enrichment",
  "source_repo": "merchant-enrichment",
  "source_file": "src/jobs/enrich_merchant.py",
  "source_lines": "112-148",
  "source_commit": "<sha>",
  "generated_at": "<iso8601>"
}
```

### 2.2 Validation records

```json
{
  "id": "val::merchant-api::mcc_format",
  "type": "validation",
  "field_id": "merchant.category_code",
  "rule": "must match ^[0-9]{4}$",
  "condition": "always",
  "layer": "api",
  "on_failure": "HTTP 400 INVALID_MCC",
  "shared_with_layers": [],
  "source_repo": "merchant-api",
  "source_file": "src/validators/MerchantValidator.java",
  "source_lines": "88-95",
  "source_commit": "<sha>",
  "generated_at": "<iso8601>"
}
```

`layer` ∈ `ingestion | cleaning | enrichment | nosql_load | api | ui`. `on_failure` ∈ `reject_record | null_out | default_value | quarantine | http_4xx | ui_inline_error | log_only`.

### 2.3 Flow edge records

```json
{
  "id": "edge::stg_merchant_raw->enrich-merchant-job",
  "type": "flow_edge",
  "from": {"kind": "dataset", "ref": "hive:stg_merchant_raw"},
  "to":   {"kind": "job",     "ref": "enrich-merchant-job"},
  "fields": ["merchant.raw_sic_code", "merchant.industry_text"],
  "source_repo": "merchant-enrichment",
  "source_commit": "<sha>"
}
```

### 2.4 Per-source extraction strategy

| Source | Strategy | Notes |
|---|---|---|
| Hive / BigQuery SQL | Parse with `sqlglot` / `sqllineage` for column-level lineage; agent summarizes logic | Flag dynamic/templated SQL as uncertain |
| Spark (PySpark/Scala) | Agent reads job code, emits records; longer term enforce a structured header block per job (declared inputs/outputs/mappings) validated in CI | Header block is the 80/20 win |
| NoSQL loaders | Parse mapping configs directly (column → CF/qualifier, column → doc path) | Usually the easiest, do early |
| API services | OpenAPI specs + DTO annotations (`@NotNull`, `@Pattern`, custom validators); agent maps response fields → field_ids using the registry | Response→source mapping may need one-time human pass |
| UI | Extract form validation (required, regex, length) from frontend or shared schema | Tag `layer: ui` |

### 2.5 Agent prompt template (per repo)

> You are extracting structured facts from the repo `<name>`. Use the canonical field registry at `registry/fields/` to resolve every raw column/field name to a `field_id`. If a name does not resolve, add it to `_unresolved_names.json` — do NOT invent a field_id. Emit `transformations/`, `validations/`, and `flow_edges/` records following the schemas in this document, with exact `source_file` and `source_lines` for every record. If logic is dynamic, config-driven, or ambiguous, put it in `_uncertain.json` with your best description and a question for the owning engineer. Do not summarize logic you have not read.

### 2.6 Coverage report
After each repo, emit `coverage_report.json`: jobs found vs. jobs extracted, fields touched vs. fields resolved to registry, count of uncertain items. **This is what tells the agent (and you) where the knowledge base is blind.**

---

## 3. Phase 3 — Rendered Documents (Layer 2)

Generated from Layer 1 + registry. These are the documents that get chunked and embedded. Templates below; every doc starts with a metadata frontmatter block:

```yaml
---
doc_type: field_card            # field_card | job | dataset | api | authored
field_ids: [merchant.category_code]
layers: [enrichment, api, ui]
repos: [merchant-enrichment, merchant-api, merchant-ui]
source_commits: {merchant-enrichment: <sha>, ...}
generated_at: <iso8601>
---
```

### 3.1 Field card (one per field_id) — the workhorse document

```markdown
# Field: merchant_category_code (merchant.category_code)

## What it is
<description + business_meaning from registry>

## Where it lives (aliases)
<table of system / container / raw name from registry>

## Validations — ALL layers
### Common (shared)
- ...
### Ingestion
- ...
### Enrichment
- ...
### API
- `^[0-9]{4}$`, on failure HTTP 400 INVALID_MCC (merchant-api, MerchantValidator.java:88)
### UI
- ...

## Lineage summary
<pre-rendered traversal from graph: source feed → cleaning job → enrichment job → BQ table → Bigtable loader → API → UI, with transformation one-liners on each hop>

## Transformations that produce or consume it
<list from transformation records>

## Owner & gotchas
<owner, notes from registry, links to authored docs>
```

This single document fully answers “what are all validation rules for field X” via plain retrieval.

### 3.2 Job doc (one per Spark job / scheduled query)
Purpose, schedule/trigger, inputs (datasets + field_ids), outputs, transformation list, failure modes, runbook link, owning team.

### 3.3 Dataset doc (one per table / keyspace / CF)
Schema table with field_id links, producers (jobs), consumers (jobs/APIs), SLA/freshness, partitioning.

### 3.4 API endpoint doc
Request/response fields → field_ids, validations, auth, backing datasets.

### 3.5 Authored docs (human-written, NOT regenerated)
- Architecture overview per process area (loading, cleaning, enrichment, NoSQL load, API/UI)
- Design decisions and the **why** (no parser can extract this — budget engineer time)
- Operational runbooks, common incidents, tribal knowledge
- Same frontmatter convention, `doc_type: authored`, link field_ids where relevant.

---

## 4. Phase 4 — Lineage Graph

Build `graph/lineage.json` from flow edges + transformations:

```json
{
  "nodes": [
    {"id": "hive:stg_merchant_raw", "kind": "dataset"},
    {"id": "enrich-merchant-job", "kind": "job"},
    {"id": "merchant.category_code", "kind": "field"}
  ],
  "edges": [
    {"from": "hive:stg_merchant_raw", "to": "enrich-merchant-job", "kind": "feeds", "fields": ["merchant.raw_sic_code"]},
    {"from": "enrich-merchant-job", "to": "merchant.category_code", "kind": "produces", "via": "txf::enrich-merchant-job::mcc_derivation"}
  ]
}
```

Expose to the agent as a tool: `get_lineage(field_id, direction=upstream|downstream|both)` → ordered path. Lineage questions are answered by traversal, never by similarity search. Also pre-render each field’s lineage into its field card (3.1) so plain RAG gets a decent answer too.

---

## 5. Quality Gates & Operating Rules

1. **Provenance everywhere** — agent answers must cite repo + file + commit. No citation, no claim.
2. **Regenerate on merge** — Layer 1 + Layer 2 rebuild in CI on merge to main of each source repo; stamp commit SHAs. Stale docs kill trust faster than missing docs.
3. **Precedence on conflict** — extracted-from-code > generated docs > authored docs > wiki. Surface conflicts, don’t silently pick.
4. **No guessing** — unresolved names and uncertain logic go to review queues (`_unresolved_names.json`, `_uncertain.json`), never into the corpus.
5. **Coverage is visible** — the answering agent reads `coverage_report.json` and says “lineage is documented through enrichment; API mapping for this field is not yet extracted” when blind.
6. **PII discipline** — registry `pii_classification` drives what the corpus includes and what the agent will discuss; build only on the approved internal platform.
7. **Eval set before agent** — collect 30–50 real engineer questions with correct answers; measure retrieval hit rate and answer correctness on every corpus iteration.

---

## 6. Execution Order (vertical slice first)

1. Stand up the `knowledge-base` repo with the directory skeleton above.
2. Pick **one** business-critical field (e.g. merchant category code).
3. Registry entry for it (Phase 1, that field only) — human reviewed.
4. Extract every transformation/validation/flow edge touching it across ALL repos (Phase 2).
5. Render its field card + the job/dataset/API docs it touches (Phase 3).
6. Build the lineage graph for it and verify `get_lineage` returns the true end-to-end path (Phase 4).
7. Test the two canonical questions against this slice:
   - “What are all validation rules for <field>?” → complete, layer-grouped answer with citations.
   - “Give me end-to-end mapping of <field>.” → full lineage path.
8. Only after the slice passes: scale field-by-field by business priority, then automate regeneration in CI.

## 7. Suggested Copilot Session Plan (per repo)

| Session | Input | Output |
|---|---|---|
| 1 | Repo + this doc | Schema inventory CSV (1.1) |
| 2 | Inventory CSVs from all systems | Alias clustering proposal (1.2) → human review |
| 3 | Repo + approved registry | Layer 1 records + uncertain/unresolved queues (2.x) |
| 4 | Layer 1 records | Rendered docs for touched fields/jobs/datasets (3.x) |
| 5 | All flow edges | lineage.json update + pre-rendered lineage in field cards (4) |

Keep sessions bounded to one repo and one phase each — review output before the next session consumes it.
