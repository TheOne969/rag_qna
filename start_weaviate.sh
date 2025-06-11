#!/bin/bash

# Remove existing container if it's already there
docker rm -f weaviate 2>/dev/null

# Run Weaviate again
docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  --name weaviate \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e DEFAULT_VECTORIZER_MODULE='none' \
  -e ENABLE_MODULES='' \
  -e GRPC_ENABLED=true \
  semitechnologies/weaviate:latest


# Wait and check readiness
echo "Waiting for Weaviate to start..."
sleep 5
curl http://127.0.0.1:8080/v1/