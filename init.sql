CREATE TYPE document_state AS ENUM ('processing', 'uploaded', 'failed');

CREATE TABLE document(
    id serial PRIMARY KEY,
    pdf_filename VARCHAR(255) NOT NULL,
    json_filename VARCHAR(255) NOT NULL,
    title VARCHAR(255),
    publication VARCHAR(255),
    year INT,
    author VARCHAR(255),
    languages VARCHAR(255),
    is_valid BOOLEAN,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_vectordb_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    current_state document_state,
    UNIQUE (pdf_filename, json_filename)
);

CREATE TABLE keyword(
    id serial PRIMARY KEY,
    keyword VARCHAR(255) NOT NULL UNIQUE,
    pdf_filename VARCHAR(255) NOT NULL
);