# Contributing

1. Open an issue describing the user-visible production problem and acceptance
   evidence.
2. Keep provider output behind structured contracts and public service facades.
3. Add or update tests for every gate, retry, hash or prompt-shape change.
4. Do not add model weights, generated projects, secrets or machine-specific
   paths.
5. Run the full offline test suite and `pipeline/release_preflight.py`.
6. For prompt/research changes, update the source lock with immutable upstream
   commits, hashes and license notes. Do not copy unlicensed guide text.

Changes that weaken proof promotion, content QA, human review, episode release
or delivery validation require explicit product-owner approval.

