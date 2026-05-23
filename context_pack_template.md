Generate context_pack_templates.yaml for this repo.

Use cross_reference_index.yaml as the routing source.

For each query route, define the final context pack structure that should be assembled before sending to the LLM.

Each context pack should include:
- route_id
- query_type
- user_question
- repo_id
- selected_primary_sources
- selected_supporting_sources
- linked_gap_refs
- linked_failure_mode_refs
- linked_log_refs
- source_freshness
- missing_context
- conflict_warnings
- confidence_hint
- answer_instructions
- expected_output_sections

Do not write Python code.
Only generate YAML and short markdown explanations.
