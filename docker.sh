#!/usr/bin/env bash
build="build"
run="run"
cpu="cpu"
gpu="gpu"
if [ -z $1 ] || !([ $1 = $build ] || [ $1 = $run ]) || [ -z $2 ] || !([ $2 = $cpu ] || [ $2 = $gpu ])
    then
        echo "Error in arguments."
        echo "Usage:"
        echo "  $0 {$build, $run} {$cpu, $gpu}"
        echo ""
        echo "Commands:"
        echo "  $build, $run: Build the Docker image or run the Docker container."
        echo "  $cpu, $gpu: Use Tensorflow for CPU or for GPU."
        exit 1
fi

# Change image name in Dockerfile
if [ $2 = $cpu ]
    then
        image="gcr.io/tensorflow/tensorflow:0.12.1"
        local_image="ganlibcpu"
        cp docker-compose-cpu.example.yml docker-compose.yml
    else
        image="gcr.io/tensorflow/tensorflow:0.12.1-gpu"
        local_image="ganlibgpu"
        cp docker-compose-gpu.example.yml docker-compose.yml
fi

cp docker/Dockerfile.example docker/Dockerfile
sed -i "s%image%$image%g" docker/Dockerfile
sed -i "s%/pwd%$PWD%g" docker-compose.yml
if [ $1 = $build ]
    then
        docker-compose -p $local_image build
    else
        docker-compose -p $local_image up -d
        docker exec -it ${local_image}_worker_1 bash
fi