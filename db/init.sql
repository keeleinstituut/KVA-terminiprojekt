CREATE TYPE document_state AS ENUM ('processing', 'uploaded', 'failed');

CREATE TABLE documents(
    id serial PRIMARY KEY,
    pdf_filename VARCHAR(255) NOT NULL,
    json_filename VARCHAR(255) NOT NULL,
    title VARCHAR(255) NOT NULL,
    publication VARCHAR(255),
    year INT,
    author VARCHAR(255),
    languages VARCHAR(255),
    url VARCHAR(255),
    is_valid BOOLEAN,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_vectordb_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    current_state document_state,
    UNIQUE (pdf_filename),
    UNIQUE (title)
);

CREATE TABLE keywords(
    id serial PRIMARY KEY,
    keyword VARCHAR(255) NOT NULL,
    document_id INTEGER NOT NULL REFERENCES public.documents (id) ON DELETE CASCADE
);

CREATE TABLE prompt_sets(
    id serial PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description VARCHAR(500),
    is_default BOOLEAN DEFAULT FALSE,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE prompts(
    id serial PRIMARY KEY,
    prompt_type VARCHAR(100) NOT NULL,
    prompt_text TEXT NOT NULL,
    description VARCHAR(500),
    prompt_set_id INTEGER REFERENCES prompt_sets(id) ON DELETE CASCADE,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (prompt_type, prompt_set_id)
);

-- Insert default prompt sets
INSERT INTO prompt_sets (name, description, is_default) VALUES
('Vaikimisi terminoloogiaanalüüs', 'Vaikimisi promptide komplekt terminoloogiliseks analüüsiks paralleelrežiimis', TRUE);

-- Insert default prompts (linked to default set)
-- Note: Only parallel extraction prompts are included (single mode removed)
INSERT INTO prompts (prompt_type, prompt_text, description, prompt_set_id) VALUES
-- Specialized prompts for parallel extraction mode
('definitions_extraction',
'**Role:** You are a terminologist extracting definitions for a keyword.

**Task:** Extract ALL CORRECT definitions for user queried term from the documents provided.

**Instructions:**
- Extract every definition found, using EXACT QUOTES from the source
- A **correct definition** follows the patterns:
1) [Term] is a [genus] that/which [differentia]
2) [Term] - [genus + differentia] (can replace term in context)
3) [genus] + [essential characteristic 1] + [essential characteristic 2] + ... 
- Before answering, check whether the output meets the conditions of a correct definition and a good usage example.
- If NO DEFINITIONS are found for the user query, defintion-like statements or definitions of synonyms and closely related terms are acceptable

**Output Format (JSON only):**
```json
{
  "definitions": [
    {"text": "exact definition text from source", "source": "Document Title", "page": 1, "url": "url if provided"}
  ]
}
```

Remember: Use exact quotes only.
If no definitions found, return {"definitions": []}',
'Specialized prompt for extracting definitions only (used in parallel mode)',
(SELECT id FROM prompt_sets WHERE name = 'Vaikimisi terminoloogiaanalüüs')),

('related_terms_extraction',
'**Role:** You are a terminologist identifying semantically related terms for a keyword.

**Task:** Extract terms that are in the documents that are semantically related to user query, consider the relations using the markers below.

**CRITICAL RULES:**
- ONLY include terms that actually appear in the provided text, if the form of the word is not sg nom, modify it to sg nom
- Do NOT invent or infer terms that are not written in the sources
- The source you cite MUST be where you found the term
- Provide as many related terms as possible
- Rank by most relevant first

**Types of relationships to look for:**
1. Hypernymy (broader): is a, type of, kind of, category of, genus in definition
2. Hyponymy (narrower): such as, including, e.g., for example, types of
3. Meronymy (parts): part of, consists of, contains, includes, component of
4. Synonymy (equivalent): also called, or, i.e., namely, aka, known as, short forms
5. Function (purpose): used for, serves to, function is, designed to, performs
6. Attribute (property): characterized by, has property, with, features
7. Agent-Action: performs, carries out, responsible for, conducts
8. Instrument: using, by means of, with the help of, through
9. Other related terms: unlike, opposed to, rather than, vs, contrary to


**Output Format (JSON only):**
```json
{
  "related_terms": [
    {"term": "term from text", "relation_type": "hyperonym|hyponym|synonym|function|attribute|agent-action|instrument|other", "source": "Document where term appears", "page": 1, "url": "url if provided"}
  ]
}
```

Remember: 
- relation_type must be one of: hyperonym|hyponym|synonym|function|attribute|agent-action|instrument|other
- ONLY include terms you can quote from the provided text
- If no related terms are explicitly mentioned, return {"related_terms": []}',
'Specialized prompt for extracting related terms only (used in parallel mode)',
(SELECT id FROM prompt_sets WHERE name = 'Vaikimisi terminoloogiaanalüüs')),

('usage_evidence_extraction',
'**Role:** You are a terminologist finding usage examples for a keyword.

**Task:** Extract GOOD passages of how the user queried term is used in context, from the documents provided.

**Instructions:**
- Extract every relevant passage found in the provided sections that shows the keyword in domain-specific usage
- The actual quote (text field) must be EXACT from the source - this is critical for citation linking
- **CRITICAL:** You must include the exact "source" (Document Title) and "page" number from the provided text chunks. Do not output "Unknown" if the source is available in the text.

A **good usage example**:
1) is drawn from real-world usage in the field
2) represents common, standard usage
3) provides enough surrounding text to understand usage
4) demonstrates the term`s meaning unambiguously
5) provides additional information complementing the definition
6) is NOT a definition

**Output Format (JSON only):**
```json
{
  "usage_evidence": [
    {
      "text": "EXACT short quote from source containing the keyword",
      "source": "Document Title",
      "page": 1,
      "url": "url if provided",
    }
  ]
}
```

CRITICAL: The "text" field must contain the EXACT wording from the document. Ensure "source" and "page" are correctly extracted. Extract ALL relevant passages found, not just a limited number.',
'Specialized prompt for extracting usage evidence only (used in parallel mode)',
(SELECT id FROM prompt_sets WHERE name = 'Vaikimisi terminoloogiaanalüüs')),

-- See Also extraction prompt
('see_also_extraction',
'**Role:** You are a terminology exploration assistant.

**Task:** Based on the keyword and document sections, suggest as many terms as are relevant to explore next.

**Instructions:**
- Suggest at least 5-10 terms that are related to the keyword
- Include terms that appear frequently in the context
- Include broader concepts, narrower specializations, and related fields
- Focus on terms that would help a terminologist understand the domain better
- Terms should be in the SAME LANGUAGE as the keyword

**Output Format (JSON only):**
```json
{
  "see_also": ["term1", "term2", "term3", "term4", "term5", "..."]
}
```

Remember: These should be terms worth exploring further. If none found, return {"see_also": []}',
'Specialized prompt for suggesting related terms to explore (used in parallel mode)',
(SELECT id FROM prompt_sets WHERE name = 'Vaikimisi terminoloogiaanalüüs')),

-- Category-specific query expansion prompts for parallel extraction mode
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
(SELECT id FROM prompt_sets WHERE name = 'Vaikimisi terminoloogiaanalüüs')),

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
(SELECT id FROM prompt_sets WHERE name = 'Vaikimisi terminoloogiaanalüüs')),

('usage_evidence_query_expansion',
'**Role:** You are a terminology search assistant expanding a user query for finding usage examples.

**Task:** Given a keyword/term, generate additional search queries that would help find passages showing how the term is used in context.

**Instructions:**
- Generate queries that focus on finding actual usage examples and contextual passages
- Include variations that might appear in example sentences or explanatory text
- Include synonyms and related terms that might appear in usage contexts
- Keep terms in the SAME LANGUAGE as the input
- Generate up to 10 additional queries
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
(SELECT id FROM prompt_sets WHERE name = 'Vaikimisi terminoloogiaanalüüs'));