# --- Configuration ---
import cv2
import numpy as np
import os
import time



# Make sure these files are in the same directory as this script.
CONFIG_FILE = 'yolov3.cfg'
WEIGHTS_FILE = 'yolov3.weights'
CLASSES_FILE = 'coco.names.txt'


"""--- Pre-run Check ---
# 
# Check if all the required model files exist before starting.
required_files = [CONFIG_FILE, WEIGHTS_FILE, CLASSES_FILE]
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print("Error: The following required files are missing from the script's directory:")
    for f in missing_files:
        print(f" - {f}")
    print("\nPlease make sure all files are in the same folder as the Python script.")
    exit()
"""

# --- Main Script ---

# Confidence threshold: minimum probability to filter weak detections.
CONF_THRESHOLD = 0.5
# Non-maximum suppression threshold: to remove overlapping bounding boxes.
NMS_THRESHOLD = 0.4

# 1. Load the YOLO model and class names
# ----------------------------------------


# ----1.a This part initializes the YOLO model and loads the class names.
print("Loading YOLO model...")
net = cv2.dnn.readNet(WEIGHTS_FILE, CONFIG_FILE)
#----


#-----1.b
""""
I praksis gjør den neste kodeblokken følgende:

    Leser inn en liste over alle mulige objekter modellen kan kjenne igjen (f.eks., "person", "bil") fra en fil.
    Finner ut nøyaktig hvilke lag i det komplekse nevralt nettverket som gir det endelige svaret.
    Genererer en unik, tilfeldig farge for hver type objekt, slik at biler kan vises med én farge og personer med en annen.
    Gir en bekreftelse på at alt er klart for å begynne bildeanalysen.
"""
with open(CLASSES_FILE, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
print("Model loaded successfully. Using CPU backend.")
#----

# 2. Initialize Webcam
# --------------------
print("Starting webcam...")
# cv2.VideoCapture(0) accesses the default webcam.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

# 3. Real-time Detection Loop
# ---------------------------
while True:
    # Read a frame from the webcam. 'ret' is a boolean that is False if no frame was captured.
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    height, width, channels = frame.shape

    # 3a. Prepare the image for the network (Preprocessing)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 3b. Process the output to find bounding boxes
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 3c. Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    # 3d. Draw the final bounding boxes on the frame
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y + 30), font, 2, color, 2)
    
    # Calculate and display FPS
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)


    # 4. Display the resulting frame
    # ------------------------------
    cv2.imshow("Webcam Detection", frame)

    # 5. Exit condition
    # -----------------
    # Wait for 1ms and check if the 'q' key was pressed.
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

# 6. Cleanup
# ----------
cap.release()
cv2.destroyAllWindows()