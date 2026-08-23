package ingestion

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

type fakePublisher struct {
	event ArticleEvent
	err   error
}

func (p *fakePublisher) Publish(_ context.Context, event ArticleEvent) error {
	p.event = event
	return p.err
}
func (p *fakePublisher) Ping(context.Context) error { return p.err }
func (p *fakePublisher) Close() error               { return nil }

func TestPublishEventNormalizesAndAcceptsValidPayload(t *testing.T) {
	publisher := &fakePublisher{}
	metrics := &Metrics{}
	service := &HTTPService{
		publisher: publisher,
		metrics:   metrics,
		logger:    slog.Default(),
		now:       func() time.Time { return time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC) },
		newID:     func() (string, error) { return "generated-id", nil },
	}
	payload := `{"article_id":"article-1","title":"Fresh chip release","category":"Technology","published_at":"2026-08-23T11:00:00Z","body":"details"}`
	request := httptest.NewRequest(http.MethodPost, "/events", strings.NewReader(payload))
	response := httptest.NewRecorder()

	service.publishEvent(response, request)

	if response.Code != http.StatusAccepted {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if publisher.event.EventID != "generated-id" || publisher.event.Category != "technology" {
		t.Fatalf("published event = %#v", publisher.event)
	}
	var body map[string]string
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["event_id"] != "generated-id" {
		t.Fatalf("body = %#v", body)
	}
}

func TestReadinessReportsDependencyFailure(t *testing.T) {
	publisher := &fakePublisher{err: errors.New("broker unavailable")}
	handler := NewHTTPHandler(publisher, nil, &Metrics{}, slog.Default())
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/ready", nil))
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d", response.Code)
	}
}
