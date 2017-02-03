#!/usr/bin/env bash
cpu="cpu"
gpu="gpu"
if [ -z $1 ] || !([ $1 = $cpu ] || [ $1 = $gpu ])
    then
        echo "Please specify if you want to run Tensorflow for CPU or GPU."
        echo "Usage:"
        echo "  $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  $cpu"
        echo "  $gpu"
        exit 1
fi
./prepare-docker-compose-yml.sh $1
if [ $1 = $cpu ]
    then
        local_image="ganlibcpu"
    else
        local_image="ganlibgpu"
fi
docker-compose -p $local_image up -d
docker exec -it ${local_image}_worker_1 bash