"""
Migrate local SQLite data to cloud PostgreSQL.
Usage: set DATABASE_URL env var, then run:
    py scripts/migrate_to_postgres.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.db import migrate_sqlite_to_postgres, init_tables, DATABASE_URL

if not DATABASE_URL:
    print("ERROR: Set DATABASE_URL environment variable first!")
    print("Example: set DATABASE_URL=postgresql://user:pass@host/db?sslmode=require")
    sys.exit(1)

print(f"Database URL: {DATABASE_URL[:40]}...")
print()

print("Step 1: Creating tables in PostgreSQL...")
init_tables()

print("\nStep 2: Migrating data from SQLite...")
migrate_sqlite_to_postgres()

print("\nDone! Your cloud database is ready.")
