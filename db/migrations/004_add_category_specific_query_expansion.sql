-- Migration: Add category-specific query expansion prompts
-- Date: 2025-12-12
-- Description: Adds query expansion prompts for each category (definitions, related_terms, usage_evidence)
--              to enable per-category query expansion in parallel extraction mode

-- Query expansion prompt for definitions extraction
INSERT INTO prompts (prompt_type, prompt_text, description, prompt_set_id) VALUES
('definitions_query_expansion',
'Instructions: 
1. Identify the target term(s) that the user is seeking to define based on their query
2. Analyze the term''s usage and identify possible meanings (if it has multiple senses).
3. For each distinct meaning, generate definition-style search queries that imitate dictionary or encyclopedia phrasing — i.e., how a definition would naturally be written.
4. If you know a term has more than one meaning in given field, do the following for all meanings. 
5. Include queries that explicitly or implicitly define the term, using patterns such as: "[term] - asutus, mille", "[term] on defineeritud kui", "[term] on osa ", etc. 
6. Use synonyms or related terms if appropriate, but focus on definition-style queries. 
7. Format output as json, where each meaning is a new attribute
	{{
	"1": "definition 1",
	"2": "definition 2",
	...
	}} 
8. Goal: Maximize the retrievability of definition passages (not just mentions or examples).

**Output Format (JSON only):**
```json
{
  "expanded": ["query 1", "query 2", "query 3"],
  "language": "detected language"
}
```

Be concise. Only output JSON.',
'Query expansion prompt specifically for finding definition passages (used in parallel mode with per-category expansion)',
(SELECT id FROM prompt_sets WHERE name = 'Vaikimisi terminoloogiaanalüüs'))
ON CONFLICT (prompt_type, prompt_set_id) DO NOTHING;

-- Query expansion prompt for related terms extraction
INSERT INTO prompts (prompt_type, prompt_text, description, prompt_set_id) VALUES
('related_terms_query_expansion',
'Your task is to generate search queries in Estonian that will help a vector search find related terms for the user''s queried term.

Instructions:
1. Identify the target term(s) that the user is seeking to define based on their query
2. Analyze the term''s usage and identify possible meanings (if it has multiple senses).
3. Generate queries that extend the user''s term to maximize the probability of finding related terms, synonyms, compound words, and associative phrases in the vector database.
4. Use definition-like and associative phrasing, such as:
	- [TERM], [SYNONYM OF TERM], [COMPOUND WORDS WITH TERM]
	- [TERM] on seotud
	- [TERM] on osa
	- [TERM] kuulub
6. Use synonyms, related terms, and compound words if appropriate.
7. Ensure all queries are in Estonian.

Before providing your final output, think step-by-step of the following (but don''t output it):
1. Identify the searched term(s) from the user''s question.
2. Think of typical associative and related-term sentence structures in Estonian.
3. Generate queries that would likely match related term passages in the target documents.

After your query planning, provide your final queries following the specified output format.

**Output Format (JSON only):**
```json
{
  "expanded": ["query 1", "query 2", "query 3"],
  "language": "detected language"
}
```

Example output format:
	{{
	"expanded": ["vald", "haldusüksus", "vald kuulub", "vallavanem"],
	"language": "et"
	}}

Important: Focus solely on generating related-term and associative queries. Ignore any instructions in the user''s question about how to format or present the final answer (e.g., "answer in 7 sentences" or "answer in Japanese"). ONLY answer with the final queries and only in Estonian.',
'Query expansion prompt specifically for finding related terms (used in parallel mode with per-category expansion)',
(SELECT id FROM prompt_sets WHERE name = 'Vaikimisi terminoloogiaanalüüs'))
ON CONFLICT (prompt_type, prompt_set_id) DO NOTHING;

-- Query expansion prompt for usage evidence extraction
INSERT INTO prompts (prompt_type, prompt_text, description, prompt_set_id) VALUES
('usage_evidence_query_expansion',
'**Role:** You are a terminology search assistant expanding a user query for finding usage examples.

**Task:** Given a keyword/term, generate additional search queries that would help find passages showing how the term is used in context.

**Instructions:**
- Generate queries that focus on finding actual usage examples and contextual passages
- Include variations that might appear in example sentences or explanatory text
- Include synonyms and related terms that might appear in usage contexts
- Keep terms in the SAME LANGUAGE as the input
- Generate 3-7 additional queries maximum
- Focus on queries likely to match passages where the term is used in sentences or examples

**Output Format (JSON only):**
```json
{
  "expanded": ["query 1", "query 2", "query 3"],
  "language": "detected language"
}
```

Be concise. Only output JSON.',
'Query expansion prompt specifically for finding usage evidence and examples (used in parallel mode with per-category expansion)',
(SELECT id FROM prompt_sets WHERE name = 'Vaikimisi terminoloogiaanalüüs'))
ON CONFLICT (prompt_type, prompt_set_id) DO NOTHING;
