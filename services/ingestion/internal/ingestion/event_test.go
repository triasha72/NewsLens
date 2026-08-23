package ingestion

import (
	"testing"
	"time"
)

func TestNormalizeSuppliesEventMetadata(t *testing.T) {
	now := time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC)
	event := ArticleEvent{
		ArticleID:   " article-1 ",
		Title:       " A useful title ",
		Category:    " Technology ",
		PublishedAt: now.Add(-time.Hour),
	}

	err := event.Normalize(now, func() (string, error) { return "event-1", nil })
	if err != nil {
		t.Fatalf("Normalize() error = %v", err)
	}
	if event.EventID != "event-1" || event.ArticleID != "article-1" {
		t.Fatalf("unexpected identifiers: %#v", event)
	}
	if event.Category != "technology" || !event.ProducedAt.Equal(now) {
		t.Fatalf("unexpected normalized event: %#v", event)
	}
}

func TestDecodeEventRejectsUnknownAndTrailingFields(t *testing.T) {
	unknown := []byte(`{"event_id":"e","article_id":"a","title":"t","category":"c","published_at":"2026-08-23T12:00:00Z","produced_at":"2026-08-23T12:00:00Z","extra":true}`)
	if _, err := DecodeEvent(unknown); err == nil {
		t.Fatal("DecodeEvent() accepted an unknown field")
	}

	trailing := []byte(`{"event_id":"e","article_id":"a","title":"t","category":"c","published_at":"2026-08-23T12:00:00Z","produced_at":"2026-08-23T12:00:00Z"} {}`)
	if _, err := DecodeEvent(trailing); err == nil {
		t.Fatal("DecodeEvent() accepted a second JSON object")
	}
}
