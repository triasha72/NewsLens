FROM golang:1.23-bookworm AS build

WORKDIR /src
COPY services/ingestion/go.mod services/ingestion/go.sum* ./
RUN go mod download
COPY services/ingestion/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/newslens-ingestion ./cmd/newslens-ingestion

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/newslens-ingestion /newslens-ingestion
EXPOSE 8080
ENTRYPOINT ["/newslens-ingestion"]
