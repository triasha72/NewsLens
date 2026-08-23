package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/triasha72/NewsLens/services/ingestion/internal/adapters"
	"github.com/triasha72/NewsLens/services/ingestion/internal/ingestion"
)

type config struct {
	mode        string
	address     string
	brokers     []string
	topic       string
	dlqTopic    string
	groupID     string
	clientID    string
	databaseURL string
	retries     int
}

func environment(name, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		return value
	}
	return fallback
}

func loadConfig() (config, error) {
	retries, err := strconv.Atoi(environment("NEWSLENS_PROCESSING_RETRIES", "3"))
	if err != nil || retries < 1 || retries > 10 {
		return config{}, errors.New("NEWSLENS_PROCESSING_RETRIES must be between 1 and 10")
	}
	configured := config{
		mode:        environment("NEWSLENS_INGESTION_MODE", "all"),
		address:     environment("NEWSLENS_HTTP_ADDRESS", ":8080"),
		brokers:     strings.Split(environment("NEWSLENS_KAFKA_BROKERS", "localhost:9092"), ","),
		topic:       environment("NEWSLENS_KAFKA_TOPIC", "news-events"),
		dlqTopic:    environment("NEWSLENS_KAFKA_DLQ_TOPIC", "news-events-dlq"),
		groupID:     environment("NEWSLENS_KAFKA_GROUP_ID", "newslens-indexers"),
		clientID:    environment("NEWSLENS_KAFKA_CLIENT_ID", "newslens-ingestion"),
		databaseURL: strings.TrimSpace(os.Getenv("NEWSLENS_DATABASE_URL")),
		retries:     retries,
	}
	if configured.mode != "api" && configured.mode != "consumer" && configured.mode != "all" {
		return config{}, errors.New("NEWSLENS_INGESTION_MODE must be api, consumer, or all")
	}
	if configured.mode != "api" && configured.databaseURL == "" {
		return config{}, errors.New("NEWSLENS_DATABASE_URL is required in consumer and all modes")
	}
	return configured, nil
}

func run(ctx context.Context, configured config, logger *slog.Logger) error {
	metrics := &ingestion.Metrics{}
	var publisher ingestion.EventPublisher
	var source ingestion.EventSource
	var deadLetter ingestion.DeadLetterPublisher
	var store ingestion.ArticleStore

	if configured.mode == "api" || configured.mode == "all" {
		publisher = adapters.NewKafkaPublisher(configured.brokers, configured.topic)
		defer publisher.Close()
	}
	if configured.mode == "consumer" || configured.mode == "all" {
		storeContext, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		postgresStore, err := adapters.NewPostgresStore(storeContext, configured.databaseURL)
		if err != nil {
			return err
		}
		store = postgresStore
		defer store.Close()

		source = adapters.NewKafkaSource(
			configured.brokers,
			configured.topic,
			configured.groupID,
			configured.clientID,
		)
		defer source.Close()
		deadLetter = adapters.NewKafkaDeadLetterPublisher(configured.brokers, configured.dlqTopic)
		defer deadLetter.Close()
	}

	server := &http.Server{
		Addr:              configured.address,
		Handler:           ingestion.NewHTTPHandler(publisher, store, metrics, logger),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
		IdleTimeout:       60 * time.Second,
	}
	result := make(chan error, 2)
	go func() {
		logger.Info("HTTP service started", "address", configured.address, "mode", configured.mode)
		result <- server.ListenAndServe()
	}()

	if source != nil && store != nil && deadLetter != nil {
		worker := ingestion.NewWorker(
			source,
			store,
			deadLetter,
			metrics,
			logger,
			configured.retries,
		)
		go func() { result <- worker.Run(ctx) }()
	}

	select {
	case <-ctx.Done():
		shutdownContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return server.Shutdown(shutdownContext)
	case err := <-result:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	}
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	configured, err := loadConfig()
	if err != nil {
		logger.Error("invalid configuration", "error", err)
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	if err := run(ctx, configured, logger); err != nil {
		logger.Error("ingestion service stopped", "error", fmt.Sprintf("%v", err))
		os.Exit(1)
	}
}
