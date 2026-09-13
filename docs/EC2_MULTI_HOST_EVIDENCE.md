# Two-host EC2 evidence run

This setup is for one limited claim: a producer, search API, and one consumer
run on one EC2 host while a second consumer runs on another EC2 host. It can
show application-tier failover. Kafka and PostgreSQL remain single-host, so it
does not claim stateful high availability.

## Before launch

- Keep the instances in the same VPC and use private addresses for Kafka and
  PostgreSQL.
- Give both instances the same security group. Allow ports `5432` and `19092`
  only from that security group itself. Allow ports `8000` and `8080` only from
  your current public IP.
- The state host uses Kafka's internal Docker listener (`kafka:9092`); the app
  host uses the private EC2 listener (`19092`). This split is required because
  Kafka clients must receive an address reachable from their own network.
- Use a fresh, random `NEWSLENS_POSTGRES_PASSWORD`; do not commit it.

## Start the state host

On the first instance, clone the repository and create a local `.env` file:

```bash
NEWSLENS_STATE_PRIVATE_IP=replace-with-this-instance-private-ip
NEWSLENS_POSTGRES_PASSWORD=replace-with-a-random-password
```

Then run:

```bash
docker compose --env-file .env -f deploy/ec2/state-host.compose.yaml up --build --detach
```

## Start the second consumer

Clone the same commit on the second instance. Its `.env` uses the first
instance's private IP and the same password. Then run:

```bash
docker compose --env-file .env -f deploy/ec2/app-host.compose.yaml up --build --detach
```

Run the soak command from the state host, with the ingestion and search URLs
pointing at `localhost`:

```bash
python scripts/realtime/run_soak_evidence.py \
  --duration-minutes 60 --events 100000 --concurrency 20 \
  --environment "AWS EC2 multi-host application tier" \
  --application-hosts 2 \
  --output-dir reports/ec2-multi-host
```

Stop the second consumer host during the recovery portion, then capture the
aggregate reports. The receipt separately records that the application tier is
multi-host and that Kafka/PostgreSQL are still single-host. It must not be
presented as a full stateful failover result.

## Teardown

Download only aggregate receipts, run `docker compose down`, then terminate
both instances in the EC2 console. Check the AWS budget afterwards.
