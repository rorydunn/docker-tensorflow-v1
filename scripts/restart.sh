#!/bin/sh
#
# This is a work in progress
# To run: sh clearcache.sh
#
# Function:
# Clears caches in a local laravel project

docker exec -it tensorflowdemo_flask_1 fuser -k 5000/tcp
docker-compose up -d
