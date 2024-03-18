#!/usr/bin/env python
#import rospy
import threading
import cv2
#from sensor_msgs.msg import Image, CompressedImage
#from cv_bridge import CvBridge, CvBridgeError
from flask import Flask, render_template, redirect, Response


# numpy and scipy
import numpy as np
from scipy.ndimage import filters

app = Flask(__name__)

#bridge = CvBridge()

cv_image = 0

def callback(data):
    global cv_image
    print("received")
    try:
        # cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        #cv2.imshow("Image window", cv_image)
        np_arr = np.fromstring(data.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        print('Original Dimensions : ',cv_image.shape)
 
        scale_percent = 30 # percent of original size
        width = int(cv_image.shape[1] * scale_percent / 100)
        height = int(cv_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(cv_image, dim, interpolation = cv2.INTER_AREA)
    except CvBridgeError as e:
        print(e)

    # d = rospy.Duration(1, 0)
    # rospy.sleep(d)


#threading.Thread(target=lambda: rospy.init_node('test_node5', disable_signals=True)).start()
#image_sub = rospy.Subscriber("axis/image_raw/compressed/", CompressedImage, callback)


### From BOSDYN examples
def image_to_opencv(image, auto_rotate=True):
    """Convert an image proto message to an openCV image."""
    num_channels = 1  # Assume a default of 1 byte encodings.
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
        extension = ".png"
    else:
        dtype = np.uint8
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            num_channels = 3
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            num_channels = 4
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            num_channels = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            num_channels = 1
            dtype = np.uint16
        extension = ".jpg"

    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            # Attempt to reshape array into a RGB rows X cols shape.
            img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_channels))
        except ValueError:
            # Unable to reshape the image data, trying a regular decode.
            #print("failed reshape")
            img = cv2.imdecode(img, -1)
    else:
        #print("test format:")
        img = cv2.imdecode(img, -1)

    if auto_rotate:
        img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])

    return img, extension


def reset_image_client(robot):
    """Recreate the ImageClient from the robot object."""
    del robot.service_clients_by_name['image']
    del robot.channels_by_authority['api.spot.robot']
    return robot.ensure_client('image')

def viewer_setup():
    # Parse args
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--image-sources', help='Get image from source(s)', action='append')
    parser.add_argument('--image-service', help='Name of the image service to query.',
                        default=ImageClient.default_service_name)
    parser.add_argument('-j', '--jpeg-quality-percent', help="JPEG quality percentage (0-100)",
                        type=int, default=50)
    parser.add_argument('-c', '--capture-delay', help="Time [ms] to wait before the next capture",
                        type=int, default=100)
    parser.add_argument(
        '--disable-full-screen',
        help="A single image source gets displayed full screen by default. This flag disables that.",
        action='store_true')
    parser.add_argument('--auto-rotate', help='rotate right and front images to be upright',
                        action='store_true')
    options = parser.parse_args(argv)

    return options

def image_viewer(options, image_client, ):
    global cv_image
    keystroke = None
    timeout_count_before_reset = 0
    while keystroke != VALUE_FOR_Q_KEYSTROKE and keystroke != VALUE_FOR_ESC_KEYSTROKE:
        try:
            start = time.time()
            images_future = image_client.get_image_async(requests, timeout=0.5)
            end = time.time()
            #print(end - start)
            while not images_future.done():
                keystroke = cv2.waitKey(25)
                print(keystroke)
                if keystroke == VALUE_FOR_ESC_KEYSTROKE or keystroke == VALUE_FOR_Q_KEYSTROKE:
                    sys.exit(1)
            images = images_future.result()

        except TimedOutError as time_err:
            if timeout_count_before_reset == 5:
                # To attempt to handle bad comms and continue the live image stream, try recreating the
                # image client after having an RPC timeout 5 times.
                _LOGGER.info("Resetting image client after 5+ timeout errors.")
                image_client = reset_image_client(robot)
                timeout_count_before_reset = 0
            else:
                timeout_count_before_reset += 1
        except Exception as err:
            _LOGGER.warning(err)
            continue
        for i in range(len(images)):
            cv_image, _ = image_to_opencv(images[i], options.auto_rotate)
            #set global image here i think cv_image
            #cv2.imshow(images[i].source.name, image)

        keystroke = cv2.waitKey(options.capture_delay)



def gen():
    global cv_image
    video = cv2.VideoCapture(0)
    '''
    options = viewer_setup(sys.argv[1:])
    
    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(options.image_service)
    requests = [
        build_image_request(source, quality_percent=options.jpeg_quality_percent)
        for source in options.image_sources
    ]

    for image_source in options.image_sources:
        cv2.namedWindow(image_source, cv2.WINDOW_NORMAL)
        if len(options.image_sources) > 1 or options.disable_full_screen:
            cv2.setWindowProperty(image_source, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
        else:
            cv2.setWindowProperty(image_source, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    '''
    while True:
        success, image = video.read()

        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # from og server
    app.run(host='0.0.0.0', port=5000, debug=False)