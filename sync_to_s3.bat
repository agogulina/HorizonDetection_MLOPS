@echo off
aws s3 sync "dataset" s3://mlops-ijellyfish-2026 --endpoint-url=https://storage.yandexcloud.net --delete