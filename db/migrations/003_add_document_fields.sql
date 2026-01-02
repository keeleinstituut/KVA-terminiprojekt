-- Migration: Add comprehensive document metadata fields
-- All fields will also be synced to Qdrant for filtering

-- Create document type enum
DO $$ BEGIN
    CREATE TYPE document_type AS ENUM (
        'legal_act',        -- õigusakt
        'educational',      -- õppematerjal
        'thesis',           -- lõputöö
        'article',          -- erialaartikkel
        'glossary',         -- erialasõnastik
        'media',            -- meedia
        'social_media',     -- sotsiaalmeedia
        'other'             -- muu
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Add new columns to documents table
ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_type document_type;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS short_name VARCHAR(100);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_translation BOOLEAN DEFAULT FALSE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS valid_until DATE;

-- Add comments
COMMENT ON COLUMN documents.document_type IS 'Dokumendi tüüp: õigusakt, õppematerjal, lõputöö, erialaartikkel, erialasõnastik, meedia, sotsiaalmeedia, muu';
COMMENT ON COLUMN documents.short_name IS 'Dokumendi lühinimi (nt "KVKS" õigusakti puhul)';
COMMENT ON COLUMN documents.is_translation IS 'Kas dokument on tõlge';
COMMENT ON COLUMN documents.valid_until IS 'Kehtiv kuni - kui NULL, siis tähtajatu; kui minevikus, siis kehtetu';

-- Update languages to be an array type for proper language codes
-- Note: This requires data migration if existing data exists
-- For now, we keep VARCHAR but document that it should contain comma-separated ISO codes
COMMENT ON COLUMN documents.languages IS 'Keelekoodid komadega eraldatult (nt "et,en,ru")';
