-- Migration: Add prompt_sets table and link existing prompts
-- Run this on existing databases to add prompt sets support

-- Step 1: Create prompt_sets table
CREATE TABLE IF NOT EXISTS prompt_sets(
    id serial PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description VARCHAR(500),
    is_default BOOLEAN DEFAULT FALSE,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Step 2: Insert default prompt set
INSERT INTO prompt_sets (name, description, is_default) 
VALUES ('Vaikimisi terminoloogiaanalüüs', 'Vaikimisi promptide komplekt terminoloogiliseks analüüsiks paralleelrežiimis', TRUE)
ON CONFLICT (name) DO NOTHING;

-- Step 3: Add prompt_set_id column to prompts table (if not exists)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'prompts' AND column_name = 'prompt_set_id'
    ) THEN
        ALTER TABLE prompts ADD COLUMN prompt_set_id INTEGER REFERENCES prompt_sets(id) ON DELETE CASCADE;
    END IF;
END $$;

-- Step 4: Link existing prompts to the default prompt set
UPDATE prompts 
SET prompt_set_id = (SELECT id FROM prompt_sets WHERE name = 'Vaikimisi terminoloogiaanalüüs')
WHERE prompt_set_id IS NULL;

-- Step 5: Drop the old unique constraint and add new one (prompt_type + prompt_set_id)
-- Note: This might fail if the constraint doesn't exist or has a different name
DO $$
BEGIN
    -- Try to drop the old constraint
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'prompts_prompt_type_key' AND table_name = 'prompts'
    ) THEN
        ALTER TABLE prompts DROP CONSTRAINT prompts_prompt_type_key;
    END IF;
    
    -- Add new composite unique constraint if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'prompts_prompt_type_prompt_set_id_key' AND table_name = 'prompts'
    ) THEN
        ALTER TABLE prompts ADD CONSTRAINT prompts_prompt_type_prompt_set_id_key UNIQUE (prompt_type, prompt_set_id);
    END IF;
EXCEPTION
    WHEN others THEN
        RAISE NOTICE 'Constraint modification skipped: %', SQLERRM;
END $$;

-- Done!
SELECT 'Migration complete. Prompt sets enabled.' as status;
