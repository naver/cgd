#!/bin/bash

function download_cub200 ()
{
    DATA_DIR='./data/CUB_200_2011'
    DATA_FILE='CUB_200_2011.tgz'

    mkdir -p data
    wget "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/$DATA_FILE" -P $DATA_DIR
    tar xf $DATA_DIR/$DATA_FILE -C $DATA_DIR --strip-components=1
    #rm $DATA_DIR/$DATA_FILE
}

function print_usage ()
{
    echo -e "Usage: $PROG_NAME COMMAND [ARGS...]"
    echo -e "\nCommands:"
    echo -e "\tcub200                  Download CUB-200-2011 dataset"
}

function main ()
{
    local data_name=$1; shift

    case $data_name in
        cub200)
            download_cub200 $@
            ;;
        *)
            print_usage
            ;;
    esac
}

main "$@"
