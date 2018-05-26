#ifndef TEXT_RECOGNITION_WORD_RECOGNIZER_HPP
#define TEXT_RECOGNITION_WORD_RECOGNIZER_HPP

#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/publisher.h>
#include <image_transport/subscriber_filter.h>
#include <image_transport/transport_hints.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <nodelet/nodelet.h>
#include <object_detection_msgs/Objects.h>
#include <object_detection_msgs/cv_conversions.hpp>
#include <ros/node_handle.h>
#include <ros/package.h>
#include <ros/publisher.h>
#include <sensor_msgs/Image.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/text.hpp>

#include <boost/filesystem/path.hpp>
#include <boost/foreach.hpp>
#include <boost/scoped_ptr.hpp>

namespace text_recognition {

class WordRecognizer : public nodelet::Nodelet {
private:
  typedef message_filters::TimeSynchronizer< sensor_msgs::Image, object_detection_msgs::Objects >
      SyncSubscriber;

public:
  WordRecognizer() {}
  virtual ~WordRecognizer() {}

private:
  virtual void onInit() {
    ros::NodeHandle &nh(getNodeHandle());
    ros::NodeHandle &pnh(getPrivateNodeHandle());

    // load parameters
    const int queue_size(pnh.param("queue_size", 10));
    republish_image_ = pnh.param("republish_image", false);
    min_probability_ = pnh.param("min_probability", 0.5);

    // create the word recognizer
    const boost::filesystem::path data_dir(
        boost::filesystem::path(ros::package::getPath("text_recognition")) / "data");
    recognizer_ = cv::text::OCRHolisticWordRecognizer::create(
        pnh.param("architecture_file", (data_dir / "dictnet_vgg_deploy.prototxt").string()),
        pnh.param("weights_file", (data_dir / "dictnet_vgg.caffemodel").string()),
        pnh.param("words_file", (data_dir / "dictnet_vgg_labels.txt").string()));

    // setup recognition result publishers
    image_transport::ImageTransport it(nh);
    if (republish_image_) {
      image_publisher_ = it.advertise("image_out", 1, true);
    }
    word_publisher_ = nh.advertise< object_detection_msgs::Objects >("words_out", 1, true);

    // detection result subscribers
    const image_transport::TransportHints default_hints;
    image_subscriber_.subscribe(it, "image_raw", 1,
                                image_transport::TransportHints(default_hints.getTransport(),
                                                                default_hints.getRosHints(), pnh));
    text_subscriber_.subscribe(nh, "texts_in", 1);

    // callback on synchronized results
    sync_subscriber_.reset(new SyncSubscriber(queue_size));
    sync_subscriber_->connectInput(image_subscriber_, text_subscriber_);
    sync_subscriber_->registerCallback(&WordRecognizer::recognize, this);
  }

  void recognize(const sensor_msgs::ImageConstPtr &image_msg,
                 const object_detection_msgs::ObjectsConstPtr &text_msg) {
    namespace odm = object_detection_msgs;

    // do nothing if no nodes subscribe messages from this nodelet
    if (image_publisher_.getNumSubscribers() == 0 && word_publisher_.getNumSubscribers() == 0) {
      return;
    }

    // extract an image from the message
    const cv_bridge::CvImageConstPtr image(cv_bridge::toCvShare(image_msg, "mono8"));
    if (!image) {
      NODELET_ERROR("image conversion error");
      return;
    }
    if (image->image.empty()) {
      NODELET_ERROR("empty image");
      return;
    }

    // recognizing texts in the image
    const odm::ObjectsPtr word_msg(new odm::Objects);
    word_msg->header = image_msg->header;
    BOOST_FOREACH (const odm::Points &contour_msg, text_msg->contours) {
      // assuming a contour is a rectangle having 4 corners
      if (contour_msg.points.size() != 4) {
        NODELET_WARN("Invalid text box not exactly having 4 corners");
        continue;
      }

      // recognize the trimed image
      cv::Mat word_image(image->image, cv::Rect(odm::toCvPoint(contour_msg.points[0]),
                                                odm::toCvPoint(contour_msg.points[2])));
      std::string word;
      std::vector< float > probability;
      recognizer_->run(word_image, word, NULL, NULL, &probability);
      if (word.empty() || probability[0] < min_probability_) {
        continue;
      }

      // pack the recognition result
      word_msg->names.push_back(word);
      word_msg->probabilities.push_back(probability[0]);
      word_msg->contours.push_back(contour_msg);
    }
    if (word_msg->names.empty()) {
      return;
    }

    // republish the image where texts are found
    if (republish_image_) {
      image_publisher_.publish(image_msg);
    }
    word_publisher_.publish(word_msg);
  }

private:
  bool republish_image_;
  double min_probability_;

  image_transport::SubscriberFilter image_subscriber_;
  message_filters::Subscriber< object_detection_msgs::Objects > text_subscriber_;
  boost::scoped_ptr< SyncSubscriber > sync_subscriber_;

  image_transport::Publisher image_publisher_;
  ros::Publisher word_publisher_;

  cv::Ptr< cv::text::OCRHolisticWordRecognizer > recognizer_;
};

} // namespace text_recognition

#endif