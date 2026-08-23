package adapters

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/triasha72/NewsLens/services/ingestion/internal/ingestion"
)

type PostgresStore struct {
	pool *pgxpool.Pool
	now  func() time.Time
}

func NewPostgresStore(ctx context.Context, databaseURL string) (*PostgresStore, error) {
	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		return nil, fmt.Errorf("configure PostgreSQL pool: %w", err)
	}
	store := &PostgresStore{pool: pool, now: time.Now}
	if err := store.Ping(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	return store, nil
}

func (s *PostgresStore) Ping(ctx context.Context) error {
	if err := s.pool.Ping(ctx); err != nil {
		return fmt.Errorf("connect to PostgreSQL: %w", err)
	}
	return nil
}

func (s *PostgresStore) Persist(
	ctx context.Context,
	event ingestion.ArticleEvent,
) (ingestion.PersistResult, error) {
	transaction, err := s.pool.Begin(ctx)
	if err != nil {
		return ingestion.PersistResult{}, fmt.Errorf("begin transaction: %w", err)
	}
	defer func() { _ = transaction.Rollback(ctx) }()

	indexedAt := s.now().UTC()
	command, err := transaction.Exec(
		ctx,
		`INSERT INTO realtime_ingestion_events (event_id, article_id, produced_at, indexed_at)
		 VALUES ($1, $2, $3, $4)
		 ON CONFLICT (event_id) DO NOTHING`,
		event.EventID,
		event.ArticleID,
		event.ProducedAt,
		indexedAt,
	)
	if err != nil {
		return ingestion.PersistResult{}, fmt.Errorf("record ingestion event: %w", err)
	}

	if command.RowsAffected() == 0 {
		if err := transaction.Commit(ctx); err != nil {
			return ingestion.PersistResult{}, fmt.Errorf("commit duplicate event: %w", err)
		}
		return ingestion.PersistResult{Duplicate: true, IndexedAt: indexedAt}, nil
	}

	_, err = transaction.Exec(
		ctx,
		`INSERT INTO realtime_articles (
			article_id, title, body, category, published_at, produced_at, indexed_at, popularity
		 ) VALUES ($1, $2, $3, $4, $5, $6, $7, 0)
		 ON CONFLICT (article_id) DO UPDATE SET
			title = EXCLUDED.title,
			body = EXCLUDED.body,
			category = EXCLUDED.category,
			published_at = EXCLUDED.published_at,
			produced_at = EXCLUDED.produced_at,
			indexed_at = EXCLUDED.indexed_at
		 WHERE realtime_articles.produced_at <= EXCLUDED.produced_at`,
		event.ArticleID,
		event.Title,
		event.Body,
		event.Category,
		event.PublishedAt,
		event.ProducedAt,
		indexedAt,
	)
	if err != nil {
		return ingestion.PersistResult{}, fmt.Errorf("upsert article: %w", err)
	}
	if err := transaction.Commit(ctx); err != nil {
		return ingestion.PersistResult{}, fmt.Errorf("commit article: %w", err)
	}
	return ingestion.PersistResult{IndexedAt: indexedAt}, nil
}

func (s *PostgresStore) Close() { s.pool.Close() }
