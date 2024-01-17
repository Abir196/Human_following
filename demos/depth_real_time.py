import cv2
import time
import os.path
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests

processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

def median(x1, y1, x2, y2):
    height = x2 - x1
    width = y2 - y1
    med = (x1 + height//2, y1 + width//2)
    return med

tracker = 'tld'
wanted_tracker = cv2.legacy.TrackerTLD_create()
depths = []

def track(wanted_tracker, output_path, streaming_url):
    tracking_win = "Object Tracking"
    cropped_win = "Tracked Region"

    cap = cv2.VideoCapture(streaming_url)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_path + '.mp4', fourcc, 20.0, (700, 700))

    frame_counter = 0
    init_box = None

    while True:
        img_resp = requests.get(streaming_url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)
        frame = cv2.resize(frame, (700, 700))

        frame_counter += 1
        start = time.perf_counter()

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        if key & 0xFF == ord("s"):
            init_box = cv2.selectROI(tracking_win, frame, fromCenter=False,
                                     showCrosshair=True)
            wanted_tracker.init(frame, init_box)

        # Check if the frame counter is odd or even, and process accordingly
        if frame_counter % 2 == 0:
            if init_box is not None:
                success, box = wanted_tracker.update(frame)

                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    print("______________________________bounding_box", (x, y), (x + w, y + h))

                    crop = frame[y:y+h, x:x+w]
                    cv2.imshow(cropped_win, crop)

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
                    cv2.destroyWindow(cropped_win)

            cv2.putText(frame, 'Tracker: {}, Frame: {}'.format(tracker, frame_counter), (30, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
            cv2.imshow(tracking_win, frame)
            end = time.perf_counter()
            print('Frame: {} , Elapsed time: {:.3f}'.format(frame_counter, end-start))
            print('=========================================')
            print("________________________these are the depths", depths)
            output.write(frame)

    cap.release()
    output.release()

track(wanted_tracker, r"D:\trials\abir_vid_test", "http://192.168.25.164:8080/shot.jpg")
cv2.destroyAllWindows()
