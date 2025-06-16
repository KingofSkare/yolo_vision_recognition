# main_simple.py
import cv2
import numpy as np
import os

# --- Configuration ---
# All paths are relative to where you run the script.
# Make sure these files are in the same directory as this script.
CONFIG_FILE = 'yolov3.cfg'
WEIGHTS_FILE = 'yolov3.weights'
CLASSES_FILE = 'coco.names.txt'
IMAGE_FILE = 'Andrea.jpg' # Your image file

# Confidence threshold: minimum probability to filter weak detections.
CONF_THRESHOLD = 0.5
# Non-maximum suppression threshold: to remove overlapping bounding boxes.
NMS_THRESHOLD = 0.4


## --- Pre-run Check ---
## Check if all the required files exist before starting.
#required_files = [CONFIG_FILE, WEIGHTS_FILE, CLASSES_FILE, IMAGE_FILE]
#missing_files = [f for f in required_files if not os.path.exists(f)]

#if missing_files:
#    print("Error: The following required files are missing from the script's directory:")
#    for f in missing_files:
#        print(f" - {f}")
#    print("\nPlease make sure all files are in the same folder as the Python script.")
#    exit()


# --- Main Script ---

# 1. Load the YOLO model and class names
# ----------------------------------------
#print("Loading YOLO model...")
# Load the neural network using OpenCV's DNN module.
net = cv2.dnn.readNet(WEIGHTS_FILE, CONFIG_FILE)

# Load the object class names.
with open(CLASSES_FILE, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get the names of the output layers. These are the layers that produce the final object detections.
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
print("Model loaded successfully.")


# 2. Load and prepare the image
# -----------------------------
print(f"Loading image: {IMAGE_FILE}")
# Load the image from the file.
img = cv2.imread(IMAGE_FILE)
# We already checked if the file exists, so no need for an 'if img is None' check here.

# Get the image's dimensions. We need these to scale the bounding boxes later.
height, width, channels = img.shape

# The network requires the image in a specific format called a 'blob'.
# - 1/255.0: Scales pixel values from 0-255 to 0-1.
# - (416, 416): Resizes the image to the size the network expects.
# - swapRB=True: OpenCV reads images in BGR format, but YOLO expects RGB. This swaps the channels.
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)


# 3. Perform the detection (Inference)
# ------------------------------------
print("Detecting objects...")
# Set the prepared image 'blob' as the input for the network.
net.setInput(blob)
# Run the detection. This is the "forward pass" that gives us the object predictions.
outs = net.forward(output_layers)
print("Detection complete.")


# 4. Process the results
# ----------------------
class_ids = []
confidences = []
boxes = []

# Loop through all the detections the network made.
for out in outs:
    for detection in out:
        # The first 5 values are box info; the rest are scores for each class.
        scores = detection[5:]
        class_id = np.argmax(scores) # Find the class with the highest score.
        confidence = scores[class_id]

        # Filter out weak detections.
        if confidence > CONF_THRESHOLD:
            # The network returns center coordinates and dimensions as percentages.
            # We scale them back to the original image's pixel dimensions.
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate the top-left corner coordinates of the bounding box.
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Store the valid detections.
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 5. Remove redundant boxes (Non-Maximum Suppression)
# ----------------------------------------------------
# An object might be detected multiple times with slightly different boxes.
# NMS keeps only the best, most confident box and suppresses the others.
indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)


# 6. Draw the final boxes on the image
# ------------------------------------
# Generate a unique color for each possible class.
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

if len(indices) > 0:
    print(f"Found {len(indices)} objects.")
    for i in indices.flatten():
        # Get the box coordinates.
        x, y, w, h = boxes[i]

        # Get the class label and confidence score.
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_ids[i]]

        # Draw the rectangle around the object.
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Create the text label.
        text = f"{label}: {confidence:.2f}"

        # Put the label on the image above the box.
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
else:
    print("No objects found with the current confidence threshold.")


# 7. Display and save the output
# ------------------------------
output_filename = 'Andrea_output.jpg'
cv2.imwrite(output_filename, img)
print(f"Output saved to '{output_filename}'")

# Try to show the image in a window.
try:
    cv2.imshow("Detection Result", img)
    print("Press any key in the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except cv2.error:
    print("\nCould not display the image in a window.")
    print(f"Please check the saved '{output_filename}' file to see the result.")