package adapters

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/segmentio/kafka-go"
	"github.com/triasha72/NewsLens/services/ingestion/internal/ingestion"
)

type KafkaPublisher struct {
	brokers []string
	writer  *kafka.Writer
}

func NewKafkaPublisher(brokers []string, topic string) *KafkaPublisher {
	return &KafkaPublisher{
		brokers: brokers,
		writer: &kafka.Writer{
			Addr:         kafka.TCP(brokers...),
			Topic:        topic,
			Balancer:     &kafka.Hash{},
			RequiredAcks: kafka.RequireAll,
			Async:        false,
			BatchSize:    100,
			BatchTimeout: 10 * time.Millisecond,
		},
	}
}

func (p *KafkaPublisher) Publish(ctx context.Context, event ingestion.ArticleEvent) error {
	payload, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("encode event: %w", err)
	}
	return p.writer.WriteMessages(ctx, kafka.Message{
		Key:   []byte(event.ArticleID),
		Value: payload,
		Time:  event.ProducedAt,
	})
}

func (p *KafkaPublisher) Ping(ctx context.Context) error {
	if len(p.brokers) == 0 {
		return errors.New("no Kafka brokers configured")
	}
	connection, err := kafka.DialContext(ctx, "tcp", p.brokers[0])
	if err != nil {
		return fmt.Errorf("connect to Kafka: %w", err)
	}
	return connection.Close()
}

func (p *KafkaPublisher) Close() error { return p.writer.Close() }

type KafkaSource struct {
	reader *kafka.Reader
}

func NewKafkaSource(brokers []string, topic, groupID, clientID string) *KafkaSource {
	return &KafkaSource{reader: kafka.NewReader(kafka.ReaderConfig{
		Brokers:           brokers,
		Topic:             topic,
		GroupID:           groupID,
		Dialer:            &kafka.Dialer{ClientID: clientID, Timeout: 10 * time.Second},
		MinBytes:          1,
		MaxBytes:          ingestion.MaxEventBytes,
		CommitInterval:    1 * time.Second,
		HeartbeatInterval: 2 * time.Second,
		SessionTimeout:    6 * time.Second,
		RebalanceTimeout:  10 * time.Second,
		StartOffset:       kafka.FirstOffset,
	})}
}

func (s *KafkaSource) Fetch(ctx context.Context) (ingestion.Message, error) {
	message, err := s.reader.FetchMessage(ctx)
	if err != nil {
		return ingestion.Message{}, err
	}
	return ingestion.Message{
		Topic:     message.Topic,
		Partition: message.Partition,
		Offset:    message.Offset,
		Key:       message.Key,
		Value:     message.Value,
		Time:      message.Time,
	}, nil
}

func (s *KafkaSource) Commit(ctx context.Context, message ingestion.Message) error {
	return s.reader.CommitMessages(ctx, kafka.Message{
		Topic:     message.Topic,
		Partition: message.Partition,
		Offset:    message.Offset,
		Key:       message.Key,
		Value:     message.Value,
		Time:      message.Time,
	})
}

func (s *KafkaSource) Lag() int64   { return s.reader.Stats().Lag }
func (s *KafkaSource) Close() error { return s.reader.Close() }

type KafkaDeadLetterPublisher struct {
	writer *kafka.Writer
}

func NewKafkaDeadLetterPublisher(brokers []string, topic string) *KafkaDeadLetterPublisher {
	return &KafkaDeadLetterPublisher{writer: &kafka.Writer{
		Addr:         kafka.TCP(brokers...),
		Topic:        topic,
		Balancer:     &kafka.Hash{},
		RequiredAcks: kafka.RequireAll,
		Async:        false,
	}}
}

func (p *KafkaDeadLetterPublisher) PublishDeadLetter(
	ctx context.Context,
	message ingestion.Message,
	reason string,
) error {
	envelope := struct {
		FailedAt              time.Time `json:"failed_at"`
		Reason                string    `json:"reason"`
		Source                string    `json:"source"`
		Partition             int       `json:"partition"`
		Offset                int64     `json:"offset"`
		OriginalPayloadBase64 []byte    `json:"original_payload_base64"`
	}{
		FailedAt:              time.Now().UTC(),
		Reason:                strings.TrimSpace(reason),
		Source:                message.Topic,
		Partition:             message.Partition,
		Offset:                message.Offset,
		OriginalPayloadBase64: message.Value,
	}
	payload, err := json.Marshal(envelope)
	if err != nil {
		return fmt.Errorf("encode dead-letter event: %w", err)
	}
	return p.writer.WriteMessages(ctx, kafka.Message{Key: message.Key, Value: payload})
}

func (p *KafkaDeadLetterPublisher) Close() error { return p.writer.Close() }
