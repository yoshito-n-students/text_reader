#ifndef TEXT_RECOGNITION_TEXT_DETECTOR_HPP
#define TEXT_RECOGNITION_TEXT_DETECTOR_HPP

#include <vector>

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
#include <opencv2/dnn.hpp>
#include <opencv2/text.hpp>

#include <boost/filesystem/path.hpp>
#include <boost/foreach.hpp>
#include <boost/function.hpp>

namespace text_recognition {

class TextDetector : public nodelet::Nodelet {
public:
  TextDetector() {}
  virtual ~TextDetector() {}

private:
  virtual void onInit() {
    ros::NodeHandle &nh(getNodeHandle());
    ros::NodeHandle &pnh(getPrivateNodeHandle());

    // load parameters
    republish_image_ = pnh.param("republish_image", false);

    // create the text detector
    const boost::filesystem::path data_dir(
        boost::filesystem::path(ros::package::getPath("text_recognition")) / "data");
    detector_ = cv::text::TextDetectorCNN::create(
        pnh.param("architecture_file", (data_dir / "textbox.prototxt").string()),
        pnh.param("weights_file", (data_dir / "TextBoxes_icdar13.caffemodel").string()));

    // create the overlapped detection suppressor
    suppressor_ = boost::bind(cv::dnn::NMSBoxes, _1, _2, pnh.param("score_threshold", 0.4),
                              pnh.param("nms_threshold", 0.5), _3, pnh.param("eta", 1.),
                              pnh.param("top_k", 0));

    // start detection
    image_transport::ImageTransport it(nh);
    const image_transport::TransportHints default_hints;
    image_subscriber_ =
        it.subscribe("image_raw", 1, &TextDetector::detect, this,
                     image_transport::TransportHints(default_hints.getTransport(),
                                                     default_hints.getRosHints(), pnh));
    if (republish_image_) {
      image_publisher_ = it.advertise("image_out", 1, true);
    }
    text_publisher_ = nh.advertise< object_detection_msgs::Objects >("texts_out", 1, true);
  }

  void detect(const sensor_msgs::ImageConstPtr &image_msg) {
    // do nothing if no nodes subscribe messages from this nodelet
    if (image_publisher_.getNumSubscribers() == 0 && text_publisher_.getNumSubscribers() == 0) {
      return;
    }

    // extract an image from the message
    cv_bridge::CvImageConstPtr image(cv_bridge::toCvShare(image_msg, "bgr8"));
    if (!image) {
      NODELET_ERROR("image conversion error");
      return;
    }
    if (image->image.empty()) {
      NODELET_ERROR("empty image");
      return;
    }

    // detect bounding boxes of texts
    std::vector< cv::Rect > boxes;
    std::vector< float > probabilities;
    detector_->detect(image->image, boxes, probabilities);
    if (boxes.empty()) {
      return;
    }

    // remove overlapped boxes
    std::vector< int > indices;
    suppressor_(boxes, probabilities, indices);
    if (indices.empty()) {
      return;
    }

    // republish the image where texts are found
    if (republish_image_) {
      image_publisher_.publish(image_msg);
    }

    // publish texts' location
    const object_detection_msgs::ObjectsPtr text_msg(new object_detection_msgs::Objects);
    text_msg->header = image_msg->header;
    BOOST_FOREACH (const int id, indices) {
      text_msg->contours.push_back(object_detection_msgs::toPointsMsg(boxes[id]));
      text_msg->probabilities.push_back(probabilities[id]);
    }
    text_publisher_.publish(text_msg);
  }

private:
  bool republish_image_;

  image_transport::Subscriber image_subscriber_;
  image_transport::Publisher image_publisher_;
  ros::Publisher text_publisher_;

  cv::Ptr< cv::text::TextDetectorCNN > detector_;
  boost::function< void(const std::vector< cv::Rect > &, const std::vector< float > &,
                        std::vector< int > &) >
      suppressor_;
};

} // namespace text_recognition

#endif