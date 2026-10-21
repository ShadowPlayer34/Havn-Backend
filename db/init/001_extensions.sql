-- Runs once, on first initialisation of an empty data volume.
-- Table DDL lives in app/bootstrap.py (SQLAlchemy metadata); this file only
-- provisions the extensions those tables depend on.

CREATE EXTENSION IF NOT EXISTS postgis;   -- geometry types, GiST, ST_* functions
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fuzzy district-name matching
