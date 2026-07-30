"""Shared process exit codes for OpenBird CLI surfaces."""

# A command opened a populated store whose stored embedding cohort differs from
# the active embedder. Recoverable by `openbird reindex`.
EXIT_REINDEX_REQUIRED = 5
