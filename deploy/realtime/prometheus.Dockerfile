FROM prom/prometheus:v3.5.0
COPY deploy/realtime/prometheus.yml /etc/prometheus/prometheus.yml
