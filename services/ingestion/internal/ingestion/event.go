package ingestion

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"
)

const MaxEventBytes = 1 << 20

type ArticleEvent struct {
	EventID     string    `json:"event_id"`
	ArticleID   string    `json:"article_id"`
	Title       string    `json:"title"`
	Category    string    `json:"category"`
	PublishedAt time.Time `json:"published_at"`
	Body        string    `json:"body"`
	ProducedAt  time.Time `json:"produced_at"`
}

func DecodeEvent(payload []byte) (ArticleEvent, error) {
	if len(payload) == 0 {
		return ArticleEvent{}, errors.New("event payload cannot be empty")
	}
	if len(payload) > MaxEventBytes {
		return ArticleEvent{}, fmt.Errorf("event payload exceeds %d bytes", MaxEventBytes)
	}

	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()

	var event ArticleEvent
	if err := decoder.Decode(&event); err != nil {
		return ArticleEvent{}, fmt.Errorf("decode event: %w", err)
	}

	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return ArticleEvent{}, errors.New("event payload must contain one JSON object")
	}

	if err := event.Validate(); err != nil {
		return ArticleEvent{}, err
	}
	return event, nil
}

func (e *ArticleEvent) Normalize(now time.Time, eventID func() (string, error)) error {
	e.EventID = strings.TrimSpace(e.EventID)
	e.ArticleID = strings.TrimSpace(e.ArticleID)
	e.Title = strings.TrimSpace(e.Title)
	e.Category = strings.ToLower(strings.TrimSpace(e.Category))
	e.Body = strings.TrimSpace(e.Body)

	if e.EventID == "" {
		generated, err := eventID()
		if err != nil {
			return fmt.Errorf("generate event id: %w", err)
		}
		e.EventID = generated
	}
	if e.ProducedAt.IsZero() {
		e.ProducedAt = now.UTC()
	}
	return e.Validate()
}

func (e ArticleEvent) Validate() error {
	checks := []struct {
		name  string
		value string
		max   int
	}{
		{"event_id", e.EventID, 128},
		{"article_id", e.ArticleID, 128},
		{"title", e.Title, 500},
		{"category", e.Category, 100},
	}

	for _, check := range checks {
		if strings.TrimSpace(check.value) == "" {
			return fmt.Errorf("%s is required", check.name)
		}
		if len(check.value) > check.max {
			return fmt.Errorf("%s exceeds %d characters", check.name, check.max)
		}
	}

	if len(e.Body) > 100_000 {
		return errors.New("body exceeds 100000 characters")
	}
	if e.PublishedAt.IsZero() {
		return errors.New("published_at is required")
	}
	if e.ProducedAt.IsZero() {
		return errors.New("produced_at is required")
	}
	return nil
}
