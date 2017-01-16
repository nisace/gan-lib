#!/usr/bin/env bash
cp docker-compose.example.yml docker-compose.yml
sed -i "s%/pwd%$PWD%g" docker-compose.yml
docker-compose up -d
docker exec -it ganlib_worker_1 bash