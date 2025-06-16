# camera.py
# This script starts the camera and displays the video feed in a window.

# Commented out parts can be used to set a custom camera size or check if the camera opened correctly.
# If commenting in the custom size make sure to comment out the default camera size part.


import cv2

#------ Start locked camera size ------
def start_camera():
    # 0 is usually the default webcam.
    cap = cv2.VideoCapture(0)

    """ ---- Uncomment this block if you want to check if the camera opened correctly ------
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    """
    
#------ Stop locked camera size ------


# #------ Start custom camera size ------
# width = 1920
# height = 1080
# def start_camera_with_custom_size(width, height):
#     # 0 is usually the default webcam.
#     cap = cv2.VideoCapture(0)

#     # if not cap.isOpened():
#     #     print("Error: Could not open video stream.")
#     #     return

#     # Attempt to set the frame width and height
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#     """ ---- Uncomment this block if you want to check if the camera opened correctly with custom size------
#     if not cap.isOpened():
#         print("Error: Could not open video stream.")
#         return
#     """

#  #------ Stop custom camera size ------

    print("Camera started. Press 'q' to quit.")

#------ Main loop to capture frames ------
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#------ End of main loop ------

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print("Camera stopped.")

if __name__ == '__main__':
    # Uncomment the line below to start the camera with default size
    start_camera()
    #start_camera_with_custom_size(width, height)