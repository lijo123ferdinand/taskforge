# Message Queues

A message queue decouples producers from consumers so each side can scale and fail independently. Producers append messages and consumers acknowledge them after processing.

Delivery guarantees matter. At-least-once delivery is the common default, which means consumers must be idempotent because the same message can arrive twice.

Backpressure protects the system when consumers fall behind. Bounded queues, consumer lag alerts and dead-letter queues keep a slow consumer from becoming an outage.
