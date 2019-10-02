#!/bin/sh
#
# This is a work in progress
#
# Run this script to delete docker containers
#
# Usage: shutdown.sh

# Remove all containers
echo "Stopping All Docker Containers"
docker stop $(docker ps -a -q)
echo "Prunning Unused Containers"
docker system prune -f
echo "Removing All Dangling Containers"
docker volume rm $(docker volume ls -qf dangling=true)
