#!/usr/bin/env bash
cpu="cpu"
if [ $1 = $cpu ]
    then
        cp docker-compose-cpu.example.yml docker-compose.yml
    else
        cp docker-compose-gpu.example.yml docker-compose.yml
fi
sed -i "s%/pwd%$PWD%g" docker-compose.yml