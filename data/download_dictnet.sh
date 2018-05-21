#!/usr/bin/env bash

# assuming this script locates a directory for dictnet data
DATA_DIR=$(builtin cd "`dirname "${BASH_SOURCE[0]}"`" > /dev/null && pwd)

wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg.caffemodel -P ${DATA_DIR}
wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_deploy.prototxt -P ${DATA_DIR}
wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_labels.txt -P ${DATA_DIR}