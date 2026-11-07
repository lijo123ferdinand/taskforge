# Rate Limiting

Rate limiting protects shared services from overload and abuse by capping how many requests a client can make. The token bucket algorithm allows short bursts while enforcing an average rate.

Limits can be enforced per user, per API key or per IP address. Distributed limiters keep counters in a shared store, which adds latency but keeps limits consistent across replicas.

Clients should see clear signals when they are limited. Returning status 429 with a retry-after header lets well-behaved clients back off instead of retrying immediately.
