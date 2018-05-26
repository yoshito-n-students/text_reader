# text_reader
A ROS nodelet to read and locate texts in an image

## Dependencies
object_detection_msgs
* https://github.com/yoshito-n-students/object_detection_msgs

## Nodelet: TextDetector
* subscribe images and detect bounding boxes of word-level texts (text boxes)

### Subscribed topics
image_raw (sensor_msgs/Image)

### Published topics
image_out (sensor_msgs/Image)
* image in which text boxes are found
* never advertised and published unless ~republish_image is ture

texts_out (object_detection_msgs/Objects)

### Parameters
~republish_image (bool, default: true)
* republish image if one text box at least is detected

~architecture_file (string, default: "$(find text_recognition)/data/textbox.prototxt")\
~weights_file (string, default: "$(find text_recognition)/data/TextBoxes_icdar13.caffemodel")
* deep neural network to detect text boxes
* valid as arguments for cv::text::TextDetectorCNN::create()

~score_threshold (double, default: 0.4)\
~nms_threshold (double, default: 0.5)\
~eta (double, default: 1.0)\
~top_k (int, default: 0)
* post-filtering to suppress overlapped detection
* valid as arguments for cv::dnn::NMSBoxes()

~image_transport (string, default: "raw")

## Nodelet: WordRecognizer
* subscribe images and corresponding text boxes, and recognize words

### Subscribed topics
image_raw (sensor_msgs/Image)

texts_in (object_detection_msgs/Objects)
* text boxes on subscribed images
* timestamp must match that of a subscribed image

### Published topics
image_out (sensor_msgs/Image)
* image in which one word at least is recognized
* never advertised and published unless ~republish_image is ture

words_out (object_detection_msgs/Objects)

### Parameters
~queue_size (int, default: 10)
* queue size of synchronized subscriber of images and text boxes

~republish_image (bool, default: false)
* republish image if one word at least is recognized

~min_probability (double, default: 0.5)
* minimum probability of published words

~architecture_file (string, default: "$(find text_recognition)/data/dictnet_vgg_deploy.prototxt")\
~weights_file (string, default: "$(find text_recognition)/data/dictnet_vgg.caffemodel")\
~words_file (string, default: "$(find text_recognition)/data/dictnet_vgg_labels.txt")
* deep neural network to recognize a word in a text box
* valid as arguments for cv::text::OCRHolisticWordRecognizer::create()

~image_transrpot (string, default: "raw")

## Eamples
see [launch/text.launch](launch/test.launch)