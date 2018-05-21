#ifndef TEXT_READER_TEXT_READER_HPP
#define TEXT_READER_TEXT_READER_HPP

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/publisher.h>
#include <image_transport/subscriber.h>
#include <image_transport/transport_hints.h>
#include <nodelet/nodelet.h>
#include <object_detection_msgs/Objects.h>
#include <object_detection_msgs/cv_conversions.hpp>
#include <ros/node_handle.h>
#include <ros/package.h>
#include <ros/publisher.h>
#include <sensor_msgs/Image.h>

#include <opencv2/core.hpp>
#include <opencv2/text.hpp>

#include <boost/filesystem/path.hpp>

namespace text_reader {

class TextReader : public nodelet::Nodelet {
public:
  TextReader() {}
  virtual ~TextReader() {}

private:
  virtual void onInit() {
    ros::NodeHandle &nh(getNodeHandle());
    ros::NodeHandle &pnh(getPrivateNodeHandle());

    // load parameters
    const boost::filesystem::path data_directory(
        boost::filesystem::path(ros::package::getPath("text_reader")) / "data");
    const std::string architecture_file(
        pnh.param("architecture_file", (data_directory / "dictnet_vgg_deploy.prototxt").string()));
    const std::string weights_file(
        pnh.param("weights_file", (data_directory / "dictnet_vgg.caffemodel").string()));
    const std::string words_file(
        pnh.param("words_file", (data_directory / "dictnet_vgg_labels.txt").string()));
    republish_image_ = pnh.param("republish_image", false);
    publish_entire_word_ = pnh.param("publish_entire_word", false);
    min_confidence_ = pnh.param("min_confidence", 0.);

    // create the text reader
    reader_ =
        cv::text::OCRHolisticWordRecognizer::create(architecture_file, weights_file, words_file);

    // start reading texts
    image_transport::ImageTransport it(nh);
    const image_transport::TransportHints default_hints;
    image_subscriber_ =
        it.subscribe("image_raw", 1, &TextReader::readTexts, this,
                     image_transport::TransportHints(default_hints.getTransport(),
                                                     default_hints.getRosHints(), pnh));
    if (republish_image_) {
      image_publisher_ = it.advertise("image_out", 1, true);
    }
    text_publisher_ = nh.advertise< object_detection_msgs::Objects >("texts_out", 1, true);
  }

  void readTexts(const sensor_msgs::ImageConstPtr &image_msg) {
    // do nothing if no nodes subscribe messages from this nodelet
    if (image_publisher_.getNumSubscribers() == 0 && text_publisher_.getNumSubscribers() == 0) {
      return;
    }

    // extract an image from the message by using toCvCopy() because the image may be modified later
    cv_bridge::CvImagePtr image(cv_bridge::toCvCopy(image_msg, "bgr8"));
    if (!image) {
      NODELET_ERROR("image conversion error");
      return;
    }
    if (image->image.empty()) {
      NODELET_ERROR("empty image");
      return;
    }

    // reading texts in the image
    std::string entire_word;
    std::vector< cv::Rect > word_rects;
    std::vector< std::string > words;
    std::vector< float > confs;
    reader_->run(image->image, entire_word, &word_rects, &words, &confs);
    if (entire_word.empty()) {
      return;
    }

    // republish the image where texts are found
    if (republish_image_) {
      image_publisher_.publish(image_msg);
    }

    // publish texts and their location
    const object_detection_msgs::ObjectsPtr texts(new object_detection_msgs::Objects);
    texts->header = image_msg->header;
    if (publish_entire_word_) {
      texts->names.push_back(entire_word);
      texts->contours.push_back(object_detection_msgs::toPointsMsg(image->image.size()));
      texts->probabilities.push_back(-1.);
    }
    for (std::size_t i = 0; i < words.size(); ++i) {
      if (confs[i] < min_confidence_) {
        continue;
      }
      texts->names.push_back(words[i]);
      texts->contours.push_back(object_detection_msgs::toPointsMsg(word_rects[i]));
      texts->probabilities.push_back(confs[i]);
    }
    text_publisher_.publish(texts);
  }

private:
  bool republish_image_, publish_entire_word_;
  double min_confidence_;

  image_transport::Subscriber image_subscriber_;
  image_transport::Publisher image_publisher_;
  ros::Publisher text_publisher_;

  cv::Ptr< cv::text::OCRHolisticWordRecognizer > reader_;
};

} // namespace text_reader

#endif