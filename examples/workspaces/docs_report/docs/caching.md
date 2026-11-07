# Caching Strategies

A cache trades memory for latency by keeping recently used data close to the reader. Read-through and write-through caches keep the cache and the store consistent at the cost of extra latency on writes.

Eviction policy decides what is lost when memory runs out. LRU works well for recency-heavy workloads, while LFU protects items that are popular over a long window.

Invalidation is the hard part. Time-to-live values bound staleness, and explicit invalidation on write keeps hot keys fresh without waiting for expiry.
