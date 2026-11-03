"""Local TiDB cluster and the tools the agent uses on it (stage 5).

- ``cluster``: start / stop a minimal ``tiup playground`` cluster, memory per component
- ``sample_data``: a small, seeded e-commerce database for the fault scenarios
- ``db``: connections (``Database`` protocol; PyMySQL for the real cluster)
- ``readonly``: read-only SQL with a statement check, plus diagnosis queries
- ``write``: write operations as proposals that run only after a confirmation
"""
