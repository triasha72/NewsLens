# Two-host EC2 evidence run

This setup is for one limited claim: a producer, search API, and one consumer
run on one EC2 host while a second consumer runs on another EC2 host. It can
show application-tier failover. Kafka and PostgreSQL remain single-host, so it
does not claim stateful high availability.

## Before launch

- Keep the instances in the same VPC and use private addresses for Kafka and
  PostgreSQL.
- Give both instances the same security group. Allow ports `5432` and `9092`
  only from that security group itself. Allow ports `8000` and `8080` only from
  your current public IP.
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
pointing at `localhost`. Stop the second consumer host during the recovery
portion, then capture the aggregate reports. Keep the resulting receipt marked
as application-tier multi-host evidence; it is not a Kafka or PostgreSQL
failover result.

## Teardown

Download only aggregate receipts, run `docker compose down`, then terminate
both instances in the EC2 console. Check the AWS budget afterwards.
