# Security policy

## Secrets

Never commit `.env`, API keys, private model URLs, cookies, access tokens or
local secret text files. Use process environment variables or a secret manager.
If a credential was ever stored in a file, delete the file from version history
and rotate the credential at the provider; adding it to `.gitignore` is not
enough.

The release preflight reports secret-risk filenames and hard-coded developer
paths without printing credential values.

## Reporting

Before a public security channel exists, report vulnerabilities privately to
the repository owner. Do not open a public issue containing credentials,
personal data, unsafe model outputs or exploitable details.

## Generated content

Operators are responsible for lawful source material, likeness/voice rights,
platform policies and AI-generated-content disclosure. MiniMax H3 model usage is
also governed by its own Community License and Acceptable Use Policy.

