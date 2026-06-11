# Source of Truth Generation — Build Instructions

Instructions for generating a two-layer knowledge corpus from project repos, intended to be executed with an AI coding agent (Copilot Agent mode / Claude) working repo-by-repo. The output feeds a RAG + lineage-graph system that answers cross-repo questions about merchant data pipelines.

---

## 0. Target Architecture (what we are producing)

```
knowledge-base/                          <- new dedicated git repo
├── registry/
│   ├── fields/                          <- Layer 0: canonical field registry (YAML, human-reviewed)
│   │   └── merchant.category_code.yaml
│   └── stages.yaml                      <- repo -> stage mapping (+ per-job overrides), human-reviewed
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
- **Canonical dataset naming.** Every dataset reference uses one fixed format per system: `hive:db.table`, `bq:project.dataset.table`, `bigtable:instance.table/column_family`, `couchbase:bucket.scope.collection`, `sftp:feed/path`, `api:METHOD /path`. Cross-repo lineage stitches itself ONLY because the repo that writes a table and the repo that reads it emit the identical dataset ref. A naming mismatch silently breaks the graph at that boundary — extractors must normalize, never improvise.
- **Every job hop is recorded, including passthrough.** A job that reads a field and writes it unchanged still gets a transformation record (`logic_kind: passthrough`). "Does job X touch field Y" must always be answerable; readability is handled at render time, never by omitting hops from the data.
- Every job and dataset carries a `stage` tag (`loading | cleansing | processing | nosql_load | api | ui`), resolved from `registry/stages.yaml`.
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

### 1.4 Stage mapping (`registry/stages.yaml`)

Small human-maintained config assigning every repo (with per-job overrides) to a process stage. ~20 lines, but it drives how every lineage answer is summarized, so it is a first-class reviewed artifact like the field registry.

```yaml
stages: [loading, cleansing, processing, nosql_load, api, ui]
repos:
  merchant-loading:     loading
  merchant-cleansing:   cleansing
  merchant-enrichment:  processing
  merchant-loaders:     nosql_load
  merchant-api:         api
  merchant-ui:          ui
overrides:                              # for repos that span stages
  merchant-loading/jobs/pre_cleanse.py: cleansing
```

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
  "input_datasets": ["bq:staging.merchant_joined"],
  "output_dataset": "bq:staging.merchant_mcc",
  "job": "enrich-merchant-job",
  "layer": "enrichment",
  "stage": "processing",
  "source_repo": "merchant-enrichment",
  "source_file": "src/jobs/enrich_merchant.py",
  "source_lines": "112-148",
  "source_commit": "<sha>",
  "generated_at": "<iso8601>"
}
```

`logic_kind` ∈ `passthrough | rename | cast | normalize | lookup | derive | aggregate | filter_only | lookup+fallback | other`.

**Passthrough rule:** emit a record for EVERY job that reads a field, even when the value is untouched (`logic_kind: passthrough`, `logic_summary` noting any row-level effect, e.g. "rows deduped on merchant_id"). Passthrough records are what make the lineage graph complete; render-time collapsing (section 4) keeps answers readable. `input_datasets`/`output_dataset` use the canonical dataset naming convention — they are what lets the traversal report the field's alias at each hop via the registry.

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
| Hive / BigQuery SQL | Agent writes a `sqlglot`/`sqllineage`-based extractor script, run in CI. Parser output is the deterministic ground truth for column-level lineage; the LLM only writes `logic_summary` and interprets dynamic SQL flagged into `_uncertain.json` | Never let the LLM free-hand lineage for parseable SQL — same input must always yield identical records |
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
<pre-rendered traversal from graph at transform resolution: transforming/validating hops shown with one-liners, consecutive passthrough hops folded as "(+N passthrough jobs in <repo>)", branches shown as a tree. Link to full-resolution view.>

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

Build `graph/lineage.json` from flow edges + transformations. Every node carries its `stage` (from `registry/stages.yaml`); validation records attach to job nodes so control points appear in path views:

```json
{
  "nodes": [
    {"id": "hive:stg_merchant_raw", "kind": "dataset", "stage": "loading"},
    {"id": "enrich-merchant-job", "kind": "job", "stage": "processing",
     "validations": ["val::enrich::mcc_nonnull"]},
    {"id": "merchant.category_code", "kind": "field"}
  ],
  "edges": [
    {"from": "hive:stg_merchant_raw", "to": "enrich-merchant-job", "kind": "feeds", "fields": ["merchant.raw_sic_code"]},
    {"from": "enrich-merchant-job", "to": "merchant.category_code", "kind": "produces", "via": "txf::enrich-merchant-job::mcc_derivation"}
  ]
}
```

### 4.1 Lineage tool with resolution

Expose to the agent as: `get_lineage(field_id, direction=upstream|downstream|both, resolution=stage|transform|full)` → ordered path (a tree when branches exist — fan-out to multiple consumers is normal and must be shown, not flattened). Lineage questions are answered by traversal, never by similarity search.

**Resolutions** — full detail is always stored; resolution only changes rendering:
- `stage` (default): one entry per process stage — "passes through 6 stages, transformed in 2".
- `transform`: within each stage, show transforming/validating hops; fold consecutive `logic_kind: passthrough` runs into one summarized entry.
- `full`: every dataset and job hop. For debugging a specific value, audits, deep onboarding.

**Collapse algorithm lives in the tool, not the LLM** (deterministic code folds, LLM narrates — LLM-side summarization is non-reproducible and occasionally drops transforming hops):
1. Group consecutive hops by node `stage`.
2. Within a group, fold runs of passthrough transformation records into `{"folded": N, "repos": [...]}`.
3. Always include the count of folded hops — hidden hops are stated ("+4 passthrough jobs"), never silently dropped.
4. Report the field's alias (from registry) at each rendered dataset hop, so renames are visible along the path.

**Resolution selection by question intent:** "where does X come from / end-to-end mapping" → stage; "what transforms X / why is this value wrong" → transform; "does job J touch X / audit / debug one stage" → full (scoped); "what breaks if I change X" → transform, downstream.

Also pre-render each field's lineage (at transform resolution) into its field card (3.1) so plain RAG gets a decent answer without the tool.

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
