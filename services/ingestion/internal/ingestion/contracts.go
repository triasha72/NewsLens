package ingestion

import (
	"context"
	"time"
)

type EventPublisher interface {
	Publish(context.Context, ArticleEvent) error
	Ping(context.Context) error
	Close() error
}

type Message struct {
	Topic     string
	Partition int
	Offset    int64
	Key       []byte
	Value     []byte
	Time      time.Time
}

type EventSource interface {
	Fetch(context.Context) (Message, error)
	Commit(context.Context, Message) error
	Lag() int64
	Close() error
}

type DeadLetterPublisher interface {
	PublishDeadLetter(context.Context, Message, string) error
	Close() error
}

type PersistResult struct {
	Duplicate bool
	IndexedAt time.Time
}

type ArticleStore interface {
	Persist(context.Context, ArticleEvent) (PersistResult, error)
	Ping(context.Context) error
	Close()
}
