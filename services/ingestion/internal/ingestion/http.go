package ingestion

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"time"
)

type HTTPService struct {
	publisher EventPublisher
	store     ArticleStore
	metrics   *Metrics
	logger    *slog.Logger
	now       func() time.Time
	newID     func() (string, error)
}

func NewHTTPHandler(publisher EventPublisher, store ArticleStore, metrics *Metrics, logger *slog.Logger) http.Handler {
	service := &HTTPService{
		publisher: publisher,
		store:     store,
		metrics:   metrics,
		logger:    logger,
		now:       time.Now,
		newID:     randomID,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", service.health)
	mux.HandleFunc("GET /ready", service.ready)
	mux.HandleFunc("GET /metrics", service.prometheus)
	mux.HandleFunc("POST /events", service.publishEvent)
	return mux
}

func randomID() (string, error) {
	value := make([]byte, 16)
	if _, err := rand.Read(value); err != nil {
		return "", err
	}
	return hex.EncodeToString(value), nil
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func (s *HTTPService) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "service": "newslens-ingestion"})
}

func ping(ctx context.Context, publisher EventPublisher, store ArticleStore) error {
	if publisher != nil {
		if err := publisher.Ping(ctx); err != nil {
			return err
		}
	}
	if store != nil {
		if err := store.Ping(ctx); err != nil {
			return err
		}
	}
	return nil
}

func (s *HTTPService) ready(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	if err := ping(ctx, s.publisher, s.store); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "not_ready", "error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func (s *HTTPService) prometheus(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	s.metrics.WritePrometheus(w)
}

func (s *HTTPService) publishEvent(w http.ResponseWriter, r *http.Request) {
	if s.publisher == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "publisher is disabled in consumer mode"})
		return
	}

	reader := http.MaxBytesReader(w, r.Body, MaxEventBytes)
	payload, err := io.ReadAll(reader)
	if err != nil {
		writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "event payload is too large"})
		return
	}

	var event ArticleEvent
	jsonDecoder := json.NewDecoder(bytes.NewReader(payload))
	jsonDecoder.DisallowUnknownFields()
	if err := jsonDecoder.Decode(&event); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	var extra any
	if err := jsonDecoder.Decode(&extra); err != io.EOF {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "event payload must contain one JSON object"})
		return
	}
	if err := event.Normalize(s.now(), s.newID); err != nil {
		writeJSON(w, http.StatusUnprocessableEntity, map[string]string{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	if err := s.publisher.Publish(ctx, event); err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			writeJSON(w, http.StatusGatewayTimeout, map[string]string{"error": "publishing timed out"})
			return
		}
		s.logger.Error("publish failed", "error", err, "event_id", event.EventID)
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "event could not be published"})
		return
	}

	s.metrics.IncPublished()
	writeJSON(w, http.StatusAccepted, map[string]string{
		"status":     "accepted",
		"event_id":   event.EventID,
		"article_id": event.ArticleID,
	})
}
