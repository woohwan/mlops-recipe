#!/usr/bin/env bash

PORT=8080

curl -d '{
    "Cylinders":8,
   "Displacement":307,
   "Horsepower":130,
   "Weight":3504,
   "Acceleration":12,
   "ModelYear":70,
   "Country":"USA"
}'\
    -H "Content-Type: application/json" \
    -X POST https://kqjuzmxag5.ap-northeast-1.awsapprunner.com/predict