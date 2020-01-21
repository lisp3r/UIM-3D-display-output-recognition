# UIM-3D-display-output-recognition

## Last robust command lines
- image `python3 ./main.py --results-dir results --in-image original_samples/sc1.png --show --model ~/down/model_1.h5`
- camera `python ./main.py --camera 0 --model ~/down/model_1.h5 --results-dir results --show`
- video `python ./main.py --in-video ~/down/Telegram\ Desktop/VID_20190827_145844_4.mp4 --results-dir results --show --model ~/down/model_1.h5`


## Things to tweak

- device.DeviceFinder and display.DisplayFinder constructor args
- helpers.py
  - `get_square_ids` - `epsilon` how much different between width and height is still square
  - `get_polygon_ids` - `epsilon` part of contour perimetr to `approxPolyDP` for 3rd argument
  - ... *To be continued*

## How it works

### Find device on image

1. Image preparation

    ```python
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_filter_size, blur_filter_size), 0)
    edge = cv2.Canny(blur, canny_th1, canny_th2, 255)
    ```

2. Find a device on the image

    Finding rectangle contours (expected external contour of device) and square childrens of that contours (internal squares contours of device).

    We are searching for contour that has a square shape and has >= 8 square childrans. Here is a device.

3. Find a display on a device image

    Finding rectangle contour with greates width + height on a display image.

4. Find numbers on a display image

    Finding a contours bigger then threshold, sorting. Stripping numbers by square contour and returning it in array.

5. Recognize numbers with neural network

