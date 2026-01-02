-- Migration: Remove legacy prompts that are no longer used
-- Date: 2025-12-18
-- Description: Removes terminology_analysis (single mode) and query_expansion (generic) prompts
--              since we now only use parallel mode with category-specific prompts

-- Remove terminology_analysis prompt (single mode - no longer used)
DELETE FROM prompts 
WHERE prompt_type = 'terminology_analysis';

-- Remove generic query_expansion prompt (replaced by category-specific ones)
DELETE FROM prompts 
WHERE prompt_type = 'query_expansion';

-- Log what was removed
DO $$
DECLARE
    removed_count INTEGER;
BEGIN
    GET DIAGNOSTICS removed_count = ROW_COUNT;
    RAISE NOTICE 'Removed % legacy prompt(s)', removed_count;
END $$;

SELECT 'Migration complete. Legacy prompts removed.' as status;
