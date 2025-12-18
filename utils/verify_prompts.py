#!/usr/bin/env python3
"""
Script to verify that all 9 expected prompts exist in the database.
"""
import os
import sys

# Add parent directory to path to import db_connection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db_connection import Connection

# Expected prompts (all 7 - parallel mode only, no generic query_expansion)
EXPECTED_PROMPTS = [
    "definitions_extraction",
    "definitions_query_expansion",
    "related_terms_extraction",
    "related_terms_query_expansion",
    "usage_evidence_extraction",
    "usage_evidence_query_expansion",
    "see_also_extraction",
]

def verify_prompts():
    """Verify all expected prompts exist in the default prompt set."""
    con = Connection(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        db=os.getenv("PG_COLLECTION"),
    )
    
    try:
        con.establish_connection()
        
        # Get default prompt set ID
        set_result = con.execute_sql(
            "SELECT id FROM prompt_sets WHERE is_default = TRUE LIMIT 1",
            [{}]
        )
        
        if not set_result.get("data"):
            print("❌ ERROR: No default prompt set found!")
            return False
        
        keys = list(set_result["keys"])
        default_set_id = dict(zip(keys, set_result["data"][0]))["id"]
        print(f"✓ Found default prompt set (ID: {default_set_id})")
        
        # Get all prompts for the default set
        prompts_result = con.execute_sql(
            "SELECT prompt_type FROM prompts WHERE prompt_set_id = :set_id",
            [{"set_id": default_set_id}]
        )
        
        if not prompts_result.get("data"):
            print("❌ ERROR: No prompts found in default set!")
            return False
        
        prompt_keys = list(prompts_result["keys"])
        found_prompts = [dict(zip(prompt_keys, row))["prompt_type"] for row in prompts_result["data"]]
        print(f"\nFound {len(found_prompts)} prompts in database:")
        for pt in sorted(found_prompts):
            print(f"  - {pt}")
        
        # Check for missing prompts
        missing = set(EXPECTED_PROMPTS) - set(found_prompts)
        extra = set(found_prompts) - set(EXPECTED_PROMPTS)
        
        print(f"\n{'='*60}")
        print(f"VERIFICATION RESULTS:")
        print(f"{'='*60}")
        
        if missing:
            print(f"\n❌ MISSING PROMPTS ({len(missing)}):")
            for pt in sorted(missing):
                print(f"  - {pt}")
        else:
            print(f"\n✅ All {len(EXPECTED_PROMPTS)} expected prompts are present!")
        
        if extra:
            print(f"\n⚠️  EXTRA PROMPTS ({len(extra)}):")
            for pt in sorted(extra):
                print(f"  - {pt}")
        
        print(f"\n{'='*60}")
        
        return len(missing) == 0
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        con.close()

if __name__ == "__main__":
    success = verify_prompts()
    sys.exit(0 if success else 1)
