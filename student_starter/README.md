# Challenge-set starter

Copy `validate_challenge.py` and use it on the final `challenge.json`:

```bash
python validate_challenge.py challenge.json
```

The default requires ten records (five matched pairs) per value of
`created_by`. To check the two-record format example:

```bash
python validate_challenge.py example_challenge.json --expected-per-member 2
```

The validator checks required fields, unique IDs, creator counts, pair balance,
shared pair metadata, public-looking source URLs, exact answer substrings and
the required representation of unanswerable examples.
