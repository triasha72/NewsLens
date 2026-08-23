package ingestion

import (
	"fmt"
	"io"
	"sync/atomic"
)

type Metrics struct {
	published       atomic.Int64
	processed       atomic.Int64
	duplicates      atomic.Int64
	invalid         atomic.Int64
	deadLettered    atomic.Int64
	processingFails atomic.Int64
	consumerLag     atomic.Int64
	freshnessMicros atomic.Int64
}

func (m *Metrics) IncPublished()       { m.published.Add(1) }
func (m *Metrics) IncProcessed()       { m.processed.Add(1) }
func (m *Metrics) IncDuplicates()      { m.duplicates.Add(1) }
func (m *Metrics) IncInvalid()         { m.invalid.Add(1) }
func (m *Metrics) IncDeadLettered()    { m.deadLettered.Add(1) }
func (m *Metrics) IncProcessingFails() { m.processingFails.Add(1) }
func (m *Metrics) SetConsumerLag(value int64) {
	if value >= 0 {
		m.consumerLag.Store(value)
	}
}
func (m *Metrics) SetFreshnessMicros(value int64) {
	if value >= 0 {
		m.freshnessMicros.Store(value)
	}
}

func metric(w io.Writer, name, help, kind string, value int64) {
	fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s %s\n%s %d\n", name, help, name, kind, name, value)
}

func (m *Metrics) WritePrometheus(w io.Writer) {
	metric(w, "newslens_ingestion_published_total", "Accepted article events.", "counter", m.published.Load())
	metric(w, "newslens_ingestion_processed_total", "Persisted article events.", "counter", m.processed.Load())
	metric(w, "newslens_ingestion_duplicates_total", "Idempotently ignored events.", "counter", m.duplicates.Load())
	metric(w, "newslens_ingestion_invalid_total", "Rejected or dead-lettered invalid events.", "counter", m.invalid.Load())
	metric(w, "newslens_ingestion_dead_letter_total", "Events written to the dead-letter topic.", "counter", m.deadLettered.Load())
	metric(w, "newslens_ingestion_failures_total", "Processing attempts that exhausted retries.", "counter", m.processingFails.Load())
	metric(w, "newslens_ingestion_consumer_lag", "Current Kafka consumer lag when available.", "gauge", m.consumerLag.Load())
	metric(w, "newslens_ingestion_index_freshness_microseconds", "Produced-to-indexed latency of the last persisted event.", "gauge", m.freshnessMicros.Load())
}
