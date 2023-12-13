import cv2 as cv
import time
import os.path
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image

processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")


def median(x1,y1,x2,y2):
   height = x2 - x1
   width = y2 - y1
   med = (x1 + height//2, y1 + width//2)
   return med

tracker = 'tld'
wanted_tracker = cv.legacy.TrackerTLD_create()
depths = []

def track(wanted_tracker, output_path, video_file):
    tracking_win = "Object Tracking"
    cropped_win = "Tracked Region"
    cap = cv.VideoCapture(video_file)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    print(output_path + '.mp4')
    output = cv.VideoWriter(output_path + '.mp4', fourcc, 20.0, (700, 700))
    
    frame_counter = 0
    init_box = None

    while True:
        r, frame = cap.read()

        frame_counter += 1
        start = time.perf_counter()
        frame = cv.resize(frame, (700, 700))

        if frame is None:
            break

        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        if key & 0xFF == ord("s"):
            init_box = cv.selectROI(tracking_win, frame, fromCenter=False,
                                    showCrosshair=True)
            wanted_tracker.init(frame, init_box)

        # Check if the frame counter is odd or even, and process accordingly
        if frame_counter % 2 == 0:
            if init_box is not None:
                success, box = wanted_tracker.update(frame)

                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv.rectangle(frame, (x, y), (x + w, y + h),
                                 (0, 255, 0), 2)
                    print("______________________________bounding_box", (x, y), (x + w, y + h))

                    crop = frame[y:y+h, x:x+w]
                    cv.imshow(cropped_win, crop)

                    image = Image.fromarray(frame)

                    # prepare image for the model
                    inputs = processor(images=image, return_tensors="pt")

                    with torch.no_grad():
                        outputs = model(**inputs)
                        predicted_depth = outputs.predicted_depth

                        # interpolate to original size
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=image.size[::-1],
                        mode="bicubic",
                        align_corners=False,
                    )

                    output_depth = prediction.squeeze().cpu().numpy()
                    med = median(x, y, x + w, y + h)
                    pixel_value = output_depth[med[0]][med[1]]
                    depths.append(pixel_value)
                else:
                    cv.destroyWindow(cropped_win)

            cv.putText(frame, 'Tracker: {}, Frame: {}'.format(tracker, frame_counter), (30, 30),
                       cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
            cv.imshow(tracking_win, frame)
            end = time.perf_counter()
            print('Frame: {} , Elapsed time: {:.3f}'.format(frame_counter, end-start))
            print('=========================================')
            print("________________________these are the depths",depths)
            output.write(frame)

track(wanted_tracker, r"D:\trials\abir_vid_test", r"D:\trials\sample.mp4")
