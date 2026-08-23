CREATE TABLE IF NOT EXISTS realtime_ingestion_events (
    event_id TEXT PRIMARY KEY,
    article_id TEXT NOT NULL,
    produced_at TIMESTAMPTZ NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS realtime_articles (
    article_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    produced_at TIMESTAMPTZ NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL,
    popularity INTEGER NOT NULL DEFAULT 0 CHECK (popularity >= 0)
);

CREATE INDEX IF NOT EXISTS realtime_articles_category_published_idx
    ON realtime_articles (category, published_at DESC);

CREATE INDEX IF NOT EXISTS realtime_articles_published_idx
    ON realtime_articles (published_at DESC);
