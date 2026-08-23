package ingestion

import (
	"context"
	"log/slog"
	"time"
)

type Worker struct {
	source     EventSource
	store      ArticleStore
	deadLetter DeadLetterPublisher
	metrics    *Metrics
	logger     *slog.Logger
	retries    int
}

func NewWorker(source EventSource, store ArticleStore, deadLetter DeadLetterPublisher, metrics *Metrics, logger *slog.Logger, retries int) *Worker {
	if retries < 1 {
		retries = 1
	}
	return &Worker{source: source, store: store, deadLetter: deadLetter, metrics: metrics, logger: logger, retries: retries}
}

func (w *Worker) Run(ctx context.Context) error {
	fetchFailures := 0
	for {
		message, err := w.source.Fetch(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			w.logger.Error("fetch failed", "error", err)
			fetchFailures++
			delay := time.Duration(min(fetchFailures, 10)) * 100 * time.Millisecond
			select {
			case <-ctx.Done():
				return nil
			case <-time.After(delay):
			}
			continue
		}
		fetchFailures = 0

		commit, err := w.process(ctx, message)
		if err != nil {
			w.logger.Error("processing failed", "error", err)
			continue
		}
		if commit {
			if err := w.source.Commit(ctx, message); err != nil {
				w.logger.Error("commit failed", "error", err)
				continue
			}
		}
		w.metrics.SetConsumerLag(w.source.Lag())
	}
}

func (w *Worker) process(ctx context.Context, message Message) (bool, error) {
	event, err := DecodeEvent(message.Value)
	if err != nil {
		w.metrics.IncInvalid()
		if dlqErr := w.deadLetter.PublishDeadLetter(ctx, message, err.Error()); dlqErr != nil {
			return false, dlqErr
		}
		w.metrics.IncDeadLettered()
		return true, nil
	}

	var result PersistResult
	for attempt := 1; attempt <= w.retries; attempt++ {
		result, err = w.store.Persist(ctx, event)
		if err == nil {
			break
		}
		if attempt < w.retries {
			delay := time.Duration(attempt*100) * time.Millisecond
			select {
			case <-ctx.Done():
				return false, ctx.Err()
			case <-time.After(delay):
			}
		}
	}

	if err != nil {
		w.metrics.IncProcessingFails()
		if dlqErr := w.deadLetter.PublishDeadLetter(ctx, message, err.Error()); dlqErr != nil {
			return false, dlqErr
		}
		w.metrics.IncDeadLettered()
		return true, nil
	}

	if result.Duplicate {
		w.metrics.IncDuplicates()
		return true, nil
	}

	w.metrics.IncProcessed()
	freshness := result.IndexedAt.Sub(event.ProducedAt)
	if freshness < 0 {
		freshness = 0
	}
	w.metrics.SetFreshnessMicros(freshness.Microseconds())
	return true, nil
}
