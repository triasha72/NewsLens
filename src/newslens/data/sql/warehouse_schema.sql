CREATE TABLE warehouse_metadata (
    schema_version VARCHAR PRIMARY KEY,
    split VARCHAR NOT NULL,
    built_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    news_sha256 VARCHAR,
    behaviors_sha256 VARCHAR
);

CREATE TABLE articles (
    news_id VARCHAR PRIMARY KEY,
    category VARCHAR NOT NULL,
    subcategory VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    abstract VARCHAR NOT NULL,
    url VARCHAR NOT NULL,
    title_entities_json VARCHAR NOT NULL,
    abstract_entities_json VARCHAR NOT NULL
);

CREATE TABLE behavior_events (
    impression_id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    event_timestamp TIMESTAMP NOT NULL,
    history_size UINTEGER NOT NULL,
    candidate_count UINTEGER NOT NULL
);

CREATE TABLE user_history (
    impression_id VARCHAR NOT NULL REFERENCES behavior_events(impression_id),
    history_position UINTEGER NOT NULL,
    news_id VARCHAR NOT NULL REFERENCES articles(news_id),
    PRIMARY KEY (impression_id, history_position)
);

CREATE TABLE candidate_interactions (
    impression_id VARCHAR NOT NULL REFERENCES behavior_events(impression_id),
    candidate_position UINTEGER NOT NULL,
    news_id VARCHAR NOT NULL REFERENCES articles(news_id),
    clicked BOOLEAN NOT NULL,
    PRIMARY KEY (impression_id, candidate_position)
);

CREATE INDEX idx_behavior_user_time
ON behavior_events(user_id, event_timestamp);

CREATE INDEX idx_candidate_news
ON candidate_interactions(news_id);

CREATE VIEW article_engagement AS
SELECT
    article.news_id,
    article.category,
    article.subcategory,
    COUNT(candidate.news_id) AS candidate_exposures,
    COALESCE(SUM(CASE WHEN candidate.clicked THEN 1 ELSE 0 END), 0) AS clicks,
    CASE
        WHEN COUNT(candidate.news_id) = 0 THEN 0.0
        ELSE SUM(CASE WHEN candidate.clicked THEN 1 ELSE 0 END)::DOUBLE
            / COUNT(candidate.news_id)
    END AS click_through_rate
FROM articles AS article
LEFT JOIN candidate_interactions AS candidate
    ON candidate.news_id = article.news_id
GROUP BY article.news_id, article.category, article.subcategory;
