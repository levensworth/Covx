#!/bin/bash
cd app
docker build -t flask_docker .
docker run -p 80:80 -it flask_docker