# Consistency Models

A consistency model defines which values a read may return after concurrent writes. Strong consistency makes a distributed store behave like a single copy, at the cost of coordination latency.

Eventual consistency lets replicas diverge briefly and converge later. It suits workloads such as counters and feeds where slightly stale reads are acceptable.

Many systems offer tunable consistency through quorum reads and writes. Choosing read and write quorums that overlap guarantees that a read observes the latest acknowledged write.
