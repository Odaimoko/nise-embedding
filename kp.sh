#!/usr/bin/env bash
p=0

until [$p==1]
do
    kill $(pgrep -f python)
done