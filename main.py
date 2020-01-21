import os
import argparse
import logging
import time
import concurrent.futures

import cv2
import numpy

import helpers
import device
import display
import numbers_finder
import nn


def future_worker(image):
    device_img = DEV_FINDER.find(image)
    display_img = DISP_FINDER.find(device_img)
    number_imgs = NUM_FINDER.find(display_img)
    numbers = [RECOGNIZER.recognize(img) for img in number_imgs]
    return device_img, display_img, number_imgs, numbers

parser = argparse.ArgumentParser(description='Find device')
source_group = parser.add_mutually_exclusive_group(required=True)
source_group.add_argument('--in-image', type=str, help='One sample image')
source_group.add_argument('--in-video', type=str, help='Input video to sample it and find device on each frame')
source_group.add_argument('--camera', type=int, help='Capture from camera number X')
parser.add_argument('--results-dir', type=str, required=True, help='Dir to put device images to')
parser.add_argument('--model', type=str, required=True, help="NN model file to load")
parser.add_argument('--show', type=bool, nargs='?', const=True, default=False, help='Show some images/videos on processing')
args = parser.parse_args()

if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.debug("Hello, %USERNAME%")

DEV_FINDER = device.DeviceFinder()
DISP_FINDER = display.DisplayFinder()
NUM_FINDER = numbers_finder.NumbersFinder()
RECOGNIZER = nn.NNRecognizer(args.model)

if args.in_image:
    logger.info('Processing %s image', args.in_image)
    image = cv2.imread(args.in_image)

    device_img = device.DeviceFinder().find(image)
    display_img = display.DisplayFinder().find(device_img)
    number_imgs = numbers_finder.NumbersFinder().find(display_img)
    numbers = [RECOGNIZER.recognize(img) for img in number_imgs]
    helpers.draw_symbols(image, numbers)
    helpers.draw_symbols(display_img, numbers)

    if not os.path.isdir(args.results_dir):
       os.makedirs(args.results_dir)
    destination = os.path.join(
        args.results_dir, os.path.basename(args.in_image))
    if args.show:
        cv2.imshow("Source", image)
        cv2.imshow("Device", device_img)
        cv2.imshow("Display", display_img)
        [cv2.imshow("Number {}".format(i), number_imgs[i]) for i in range(0,len(number_imgs))]
        logger.info("Result numbers: %s", numbers)
        cv2.waitKey()
    cv2.imwrite(destination, display_img)
    logger.info("Result saved to {}".format(destination))


elif args.in_video:
    logger.info('Processing %s video', args.in_video)
    capture = cv2.VideoCapture(args.in_video)

    if not capture.isOpened():
        raise RuntimeError("Unable to open {} source".format(args.in_video))
 
    video_size = (int(capture.get(3)), int(capture.get(4)))
    video_fps = capture.get(cv2.CAP_PROP_FPS)

    out_videos = {
        'device': cv2.VideoWriter(
            os.path.join(args.results_dir, "{}_device.avi".format(os.path.basename(args.in_video))),
            cv2.VideoWriter_fourcc('M','J','P','G'),
            10,
            video_size),
        'display': cv2.VideoWriter(
            os.path.join(args.results_dir, "{}_display.avi".format(os.path.basename(args.in_video))),
            cv2.VideoWriter_fourcc('M','J','P','G'),
            10,
            video_size)
    }

    logger.debug("Start processing frames")
    frame_read_res = True
    finder_future = None
    display_found = device_found = False
    skipped_frames = 0
    dev_img = disp_img = num_imgs = numbers = None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        while(frame_read_res):
            loop_stat = time.time()
            frame_read_res, frame = capture.read()
            if frame_read_res:
                # Previous thread is still running
                if finder_future:
                    # Check it's alive 
                    if finder_future.done():
                        try:
                            dev_img, disp_img, num_imgs, numbers = finder_future.result()
                        except helpers.GeneralRecognitionFailure:
                            skipped_frames += 1
                            finder_future = None
                        finally:
                            if type(dev_img) == numpy.ndarray:
                                device_found = True
                                dev_img = cv2.resize(dev_img, video_size)
                            if type(disp_img) == numpy.ndarray:
                                display_found = True
                                disp_img = cv2.resize(disp_img, video_size)
                            if args.show:
                                if type(dev_img) == numpy.ndarray:
                                    cv2.imshow("Device", dev_img)
                                if type(disp_img) == numpy.ndarray:
                                    cv2.imshow("Display", disp_img)
                                if num_imgs:
                                    [cv2.imshow("Number {}".format(i), num_imgs[i]) for i in range(0,len(num_imgs))]
                                if numbers:
                                    logger.info("Numbers: %s", numbers)
                            logger.warning("Frames skipped: {}".format(skipped_frames))
                            logger.debug("Wasted: {}".format(time.time() - finding_start_time))
                            finding_start_time = time.time()

                            #finder_future = executor.submit(
                            #    find_all_displays_on, frame, name=os.path.basename(args.in_video))
                            skipped_frames = 0
                            finder_future = None
                    else:
                        # Thread in work, wait till done
                        skipped_frames += 1
                else:
                    finding_start_time = time.time()
                    # First time display found on screen
                    finder_future = executor.submit(future_worker, frame)
                    skipped_frames = 0
                if args.show:
                    if numbers:
                        helpers.draw_symbols(frame, numbers)
                    cv2.imshow("Video {}".format(args.in_video), frame)
            if device_found:
                out_videos['device'].write(cv2.resize(dev_img, video_size))
            if display_found:
                out_videos['display'].write(cv2.resize(disp_img, video_size))
            loop_end = time.time()
            # Count delay as for real playing input video
            original_video_delay = int(1000/video_fps)
            spent_in_loop = int((loop_end - loop_stat)*1000)
            if spent_in_loop >= original_video_delay:
                logger.error("Spent in loop more than frame delay! Do some!")
                delay = 1
            else:
                delay = original_video_delay - spent_in_loop
            cv2.waitKey(delay)
    capture.release()
    for _, output in out_videos.iteritems():
        output.release()

elif args.camera != None:
    logger.info('Capturing from %s camera', args.camera)
    capture = cv2.VideoCapture(args.camera)

    if not capture.isOpened():
        raise RuntimeError("Unable to open {} source".format(args.in_video))
 
    video_size = (int(capture.get(3)), int(capture.get(4)))
    out_videos = {
        'device': cv2.VideoWriter(
            os.path.join(args.results_dir, "camera_{}_device.avi".format(args.camera)),
            cv2.VideoWriter_fourcc('M','J','P','G'),
            10,
            video_size),
        'display': cv2.VideoWriter(
            os.path.join(args.results_dir, "camera_{}_display.avi".format(args.camera)),
            cv2.VideoWriter_fourcc('M','J','P','G'),
            10,
            video_size)
    }

    logger.debug("Start processing frames")
    dev_img = disp_img = num_imgs = numbers = None
    while(True):
        cv2.waitKey(100)
        ret, frame = capture.read()
        if ret:
            if args.show:
                cv2.imshow("Camera {}".format(args.camera), frame)
            try:
                dev_img, disp_img, num_imgs, numbers = future_worker(frame)
            except helpers.GeneralRecognitionFailure as exc:
                logger.warning("Failed to recognize values: %s, continue", exc)
                continue
            finally:
                if args.show:
                    if type(dev_img) == numpy.ndarray:
                        cv2.imshow("Device", dev_img)
                        out_videos['device'].write(cv2.resize(dev_img, video_size))
                    if type(disp_img) == numpy.ndarray:
                        cv2.imshow("Display", disp_img)
                        out_videos['display'].write(cv2.resize(disp_img, video_size))
                    if num_imgs:
                        [cv2.imshow("Number {}".format(i), num_imgs[i]) for i in range(0,len(num_imgs))]
                    if numbers:
                        logger.info("Numbers: %s", numbers)
        else:
            break

    # When everything done, release the video capture and video write objects
    capture.release()
    for _, output in out_videos.iteritems():
        output.release()
