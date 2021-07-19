#!/bin/bash
echo "killing old docker processes"
docker-compose stop
docker-compose rm -f


echo "building docker containers"
docker-compose up --build -d

