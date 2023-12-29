import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as transforms
from utils import cellboxes_to_boxes, non_max_suppression, plot_image, load_checkpoint
from model import Yolov1
import requests  # Import the requests library

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the YOLOv1 model
model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
optimizer = optim.Adam(
    model.parameters(), lr=2e-5, weight_decay=0
)
load_checkpoint(torch.load("finaltransform.pth.tar"), model, optimizer)
model.eval()

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define a transformation to preprocess the frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])
box_coordinates_list = []
frame_number = 0  # Initialize frame number

# Server URL
server_url = "http://127.0.0.1:5000/update"

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is read correctly
    if not ret:
        print("Error: Could not read frame.")
        break

    # Increment frame number
    frame_number += 1

    # Preprocess the frame
    input_tensor = transform(frame).unsqueeze(0).to(DEVICE)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Post-process the output
    bboxes = cellboxes_to_boxes(output)
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
    box_coordinates_list = []
    # Store box coordinates
    box_coordinates_list.append({
        'frame_no': frame_number,
        'data': [{
            'class_label': box[0],
            'confidence': box[1],
            'x_midpoint': box[2],
            'y_midpoint': box[3],
            'width': box[4],
            'height': box[5],
        } for box in bboxes],
    })

    im = np.array(frame)
    height, width, _ = im.shape

    for box in bboxes:
        class_label, confidence, x_midpoint, y_midpoint, box_width, box_height = box

        upper_left_x = int((x_midpoint - box_width / 2) * width)
        upper_left_y = int((y_midpoint - box_height / 2) * height)
        box_width = int(box_width * width)
        box_height = int(box_height * height)

        # Draw rectangle on the image
        cv2.rectangle(frame, (upper_left_x, upper_left_y), (upper_left_x + box_width, upper_left_y + box_height),
                      (0, 255, 0), 2)

        # Display information
        info_text = f"Class: {class_label}, Confidence: {confidence:.2f}"
        cv2.putText(frame, info_text, (upper_left_x, upper_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    2)

    # Display the image with bounding boxes
    cv2.imshow("Bounding Boxes", frame)
    cv2.waitKey(1)

    # Prepare and send data to the server
    data = {
        'frames': box_coordinates_list,
    }

    response = requests.post(server_url, json=data)

    if response.status_code == 200:
        print("Data sent successfully.")
    else:
        print(f"Error sending data. Status code: {response.status_code}")
        print(response.text)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()