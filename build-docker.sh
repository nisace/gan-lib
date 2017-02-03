#!/usr/bin/env bash
cpu="cpu"
gpu="gpu"
if [ -z $1 ] || !([ $1 = $cpu ] || [ $1 = $gpu ])
    then
        echo "Please specify if you want to build Tensorflow for CPU or GPU."
        echo "Usage:"
        echo "  $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  $cpu"
        echo "  $gpu"
        exit 1
fi

# Change image name in Dockerfile
if [ $1 = $cpu ]
    then
        image="gcr.io/tensorflow/tensorflow"
        local_image="ganlibcpu"
    else
        image="gcr.io/tensorflow/tensorflow:latest-gpu"
        local_image="ganlibgpu"
fi
cp docker/Dockerfile.example docker/Dockerfile
sed -i "s%image%$image%g" docker/Dockerfile
./prepare-docker-compose-yml.sh $1
docker-compose -p $local_image build

#docker-compose -p $local_image up -d
#docker exec -it ganlib_worker_1 bash