# Retries and Timeouts

Every network call needs a timeout, because a call without one can hold resources forever. Timeouts should be set from measured latency percentiles rather than guessed.

Retries recover from transient failures but can amplify an outage. Exponential backoff with jitter spreads retries out so recovering services are not flooded.

Only idempotent operations are safe to retry blindly. Idempotency keys let a server recognise a repeated request and return the original result.
