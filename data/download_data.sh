#!/usr/bin/env bash

# assuming this script locates a directory for dnn data
DATA_DIR=$(builtin cd "`dirname "${BASH_SOURCE[0]}"`" > /dev/null && pwd)

# text box detector data
wget https://github.com/opencv/opencv_contrib/raw/master/modules/text/samples/textbox.prototxt -P ${DATA_DIR}
wget https://www.dropbox.com/s/g8pjzv2de9gty8g/TextBoxes_icdar13.caffemodel?dl=0 -O ${DATA_DIR}/TextBoxes_icdar13.caffemodel

# word recognizer data
wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg.caffemodel -P ${DATA_DIR}
wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_deploy.prototxt -P ${DATA_DIR}
wget http://nicolaou.homouniversalis.org/assets/vgg_text/dictnet_vgg_labels.txt -P ${DATA_DIR}