package ingestion

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"testing"
	"time"
)

type fakeSource struct{ lag int64 }

func (s *fakeSource) Fetch(context.Context) (Message, error) { return Message{}, nil }
func (s *fakeSource) Commit(context.Context, Message) error  { return nil }
func (s *fakeSource) Lag() int64                             { return s.lag }
func (s *fakeSource) Close() error                           { return nil }

type fakeStore struct {
	result PersistResult
	err    error
	calls  int
}

func (s *fakeStore) Persist(context.Context, ArticleEvent) (PersistResult, error) {
	s.calls++
	return s.result, s.err
}
func (s *fakeStore) Ping(context.Context) error { return s.err }
func (s *fakeStore) Close()                     {}

type fakeDLQ struct {
	calls  int
	reason string
}

func (d *fakeDLQ) PublishDeadLetter(_ context.Context, _ Message, reason string) error {
	d.calls++
	d.reason = reason
	return nil
}
func (d *fakeDLQ) Close() error { return nil }

func testEventPayload() []byte {
	return []byte(`{"event_id":"event-1","article_id":"article-1","title":"News","category":"technology","published_at":"2026-08-23T11:00:00Z","produced_at":"2026-08-23T12:00:00Z"}`)
}

func TestWorkerProcessesAndCountsDuplicateEvents(t *testing.T) {
	indexedAt := time.Date(2026, 8, 23, 12, 0, 1, 0, time.UTC)
	store := &fakeStore{result: PersistResult{Duplicate: true, IndexedAt: indexedAt}}
	metrics := &Metrics{}
	worker := NewWorker(&fakeSource{}, store, &fakeDLQ{}, metrics, slog.Default(), 3)

	commit, err := worker.process(context.Background(), Message{Value: testEventPayload()})
	if err != nil || !commit {
		t.Fatalf("process() = (%v, %v)", commit, err)
	}
	var output bytes.Buffer
	metrics.WritePrometheus(&output)
	if !bytes.Contains(output.Bytes(), []byte("newslens_ingestion_duplicates_total 1")) {
		t.Fatalf("metrics = %s", output.String())
	}
}

func TestWorkerDeadLettersInvalidAndExhaustedEvents(t *testing.T) {
	dlq := &fakeDLQ{}
	metrics := &Metrics{}
	worker := NewWorker(&fakeSource{}, &fakeStore{}, dlq, metrics, slog.Default(), 2)
	commit, err := worker.process(context.Background(), Message{Value: []byte("not-json")})
	if err != nil || !commit || dlq.calls != 1 {
		t.Fatalf("invalid process = (%v, %v), dlq calls = %d", commit, err, dlq.calls)
	}

	store := &fakeStore{err: errors.New("database unavailable")}
	worker = NewWorker(&fakeSource{}, store, dlq, metrics, slog.Default(), 2)
	commit, err = worker.process(context.Background(), Message{Value: testEventPayload()})
	if err != nil || !commit || store.calls != 2 || dlq.calls != 2 {
		t.Fatalf("failed process = (%v, %v), store calls = %d, dlq calls = %d", commit, err, store.calls, dlq.calls)
	}
}
