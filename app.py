# import cv2
# import pytesseract
# from PIL import Image, ImageDraw
# import numpy as np
# import os
# from datetime import datetime
# import streamlit as st

# # Ensure the output folder exists
# output_folder = "wrong_frames"
# os.makedirs(output_folder, exist_ok=True)

# # Define the conditions for the letters
# conditions = {
#     'A': lambda value: value < 10,
#     'B': lambda value: value < 55,
#     'C': lambda value: value > 25,
#     'D': lambda value: value > 18
# }

# # Function to process each frame and draw bounding boxes
# def process_frame(frame):
#     # Dictionary to store detected values for the current frame
#     detected_values = {'A': None, 'B': None, 'C': None, 'D': None}

#     # Convert the captured frame to a PIL image
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?/~`'

#     # Perform OCR on the image and get bounding box information
#     data = pytesseract.image_to_boxes(image, config=custom_config)

#     # Draw bounding boxes
#     draw = ImageDraw.Draw(image)
#     width, height = image.size

#     # Parse OCR data
#     ocr_results = data.splitlines()

#     # Group characters by their y-coordinates and find the corresponding values
#     expressions = {}
#     for box in ocr_results:
#         b = box.split(' ')
#         char = b[0]
#         x1, y1, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#         y1, y2 = height - y1, height - y2

#         # If the character is a target letter, find its corresponding value
#         if char in conditions:
#             # Initialize the expression with the bounding box
#             expressions[char] = {'bbox': [x1, y2, x2, y1], 'value': ''}

#         # Find the closest numeric value to the right of the letter
#         elif char.isdigit() or char == '=':
#             if expressions:
#                 last_char = list(expressions.keys())[-1]
#                 expressions[last_char]['value'] += char
#                 expressions[last_char]['bbox'][2] = max(expressions[last_char]['bbox'][2], x2)
#                 expressions[last_char]['bbox'][3] = min(expressions[last_char]['bbox'][3], y1)
#                 expressions[last_char]['bbox'][1] = max(expressions[last_char]['bbox'][1], y2)

#     # Evaluate conditions and draw bounding boxes around expressions
#     condition_met = True
#     for char, info in expressions.items():
#         if '=' in info['value']:
#             value = info['value'].split('=')[1].strip()
#             if value.isdigit() and char in conditions:
#                 value = int(value)
#                 x1, y1, x2, y2 = info['bbox']

#                 # Store recognized character and value in detected_values dictionary
#                 detected_values[char] = value

#                 if not conditions[char](value):
#                     condition_met = False
#                     draw.rectangle([x1, y1, x2, y2], outline="red")
#                 else:
#                     draw.rectangle([x1, y1, x2, y2], outline="green")

#     # Convert the image back to OpenCV format
#     return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), detected_values


# # Streamlit application setup
# st.title('Real-time Video Processing with OCR')
# st.text('Choose between uploading a video, using your camera, or streaming via IP Webcam.')

# # Option to choose between file upload, camera, or IP Webcam
# option = st.selectbox("Choose input method", ["Upload Video", "Use Camera", "IP Webcam"])

# # Use camera for real-time processing
# if option == "Use Camera":
#     st.text("Using camera for real-time video feed. Press 'q' to stop.")
#     cap = cv2.VideoCapture(0)  # Use the camera (index 0 for default camera)
#     frame_window = st.image([])  # Placeholder for displaying frames

#     if cap.isOpened():
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frame_count = 0

#         while True:
#             ret, frame = cap.read()  # Capture frame-by-frame
#             if not ret:
#                 st.warning("Camera feed not available.")
#                 break

#             # Process one frame per second based on the camera's frame rate
#             if frame_count % int(fps) == 0:
#                 # Process the frame with OCR and draw bounding boxes
#                 processed_frame, detected_values = process_frame(frame)

#                 # Display the frame in real-time with bounding boxes
#                 frame_window.image(processed_frame, channels='BGR')

#                 # Display the detected values in the console (optional)
#                 print(f"Frame {frame_count // int(fps) + 1}: {detected_values}")

#             frame_count += 1

#             # Stop the feed if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

# elif option == "Upload Video":
#     # Upload video file
#     uploaded_file = st.file_uploader("Choose a video file", type=["wmv", "mp4", "avi"])

#     if uploaded_file is not None:
#         video_path = uploaded_file.name

#         # Save uploaded file temporarily
#         with open(video_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         # Open the video file
#         cap = cv2.VideoCapture(video_path)

#         if cap.isOpened():
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             frame_window = st.image([])  # Placeholder for displaying frames
#             frame_count = 0

#             while True:
#                 ret, frame = cap.read()  # Capture frame-by-frame
#                 if not ret:
#                     break

#                 # Process one frame per second based on the video's frame rate
#                 if frame_count % int(fps) == 0:
#                     # Process the frame with OCR and draw bounding boxes
#                     processed_frame, detected_values = process_frame(frame)

#                     # Display the frame with bounding boxes in Streamlit
#                     frame_window.image(processed_frame, channels='BGR')

#                     # Display the detected values in the console (optional)
#                     print(f"Frame {frame_count // int(fps) + 1}: {detected_values}")

#                 frame_count += 1

#             cap.release()

# # Add IP Webcam option
# elif option == "IP Webcam":
#     st.text("Stream video via IP Webcam. Enter the stream URL from your phone (e.g., http://192.168.1.x:8080/video).")
    
#     ip_webcam_url = st.text_input("Enter the IP Webcam URL:", "http://192.168.1.x:8080/video")
    
#     if st.button("Start IP Webcam Stream"):
#         st.text("Streaming IP Webcam video feed...")
#         cap = cv2.VideoCapture(ip_webcam_url)

#         if cap.isOpened():
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             frame_window = st.image([])  # Placeholder for displaying frames
#             frame_count = 0

#             while True:
#                 ret, frame = cap.read()  # Capture frame-by-frame
#                 if not ret:
#                     st.warning("IP Webcam feed not available.")
#                     break

#                 # Process one frame per second based on the video's frame rate
#                 if frame_count % int(fps) == 0:
#                     # Process the frame with OCR and draw bounding boxes
#                     processed_frame, detected_values = process_frame(frame)

#                     # Display the frame in real-time with bounding boxes
#                     frame_window.image(processed_frame, channels='BGR')

#                     # Display the detected values in the console (optional)
#                     print(f"Frame {frame_count // int(fps) + 1}: {detected_values}")

#                 frame_count += 1

#             cap.release()


import cv2
import pytesseract
from PIL import Image, ImageDraw
import numpy as np
import os
import streamlit as st

# Ensure the output folder exists
output_folder = "wrong_frames"
os.makedirs(output_folder, exist_ok=True)

# Define the conditions for the letters
conditions = {
    'A': lambda value: value < 10,
    'B': lambda value: value < 55,
    'C': lambda value: value > 25,
    'D': lambda value: value > 18
}

# Function to process each frame and draw bounding boxes
def process_frame(frame):
    detected_values = {'A': None, 'B': None, 'C': None, 'D': None}
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?/~`'
    data = pytesseract.image_to_boxes(image, config=custom_config)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    ocr_results = data.splitlines()
    expressions = {}
    
    for box in ocr_results:
        b = box.split(' ')
        char = b[0]
        x1, y1, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        y1, y2 = height - y1, height - y2

        if char in conditions:
            expressions[char] = {'bbox': [x1, y2, x2, y1], 'value': ''}
        elif char.isdigit() or char == '=':
            if expressions:
                last_char = list(expressions.keys())[-1]
                expressions[last_char]['value'] += char
                expressions[last_char]['bbox'][2] = max(expressions[last_char]['bbox'][2], x2)
                expressions[last_char]['bbox'][1] = max(expressions[last_char]['bbox'][1], y2)

    for char, info in expressions.items():
        if '=' in info['value']:
            value = info['value'].split('=')[1].strip()
            if value.isdigit() and char in conditions:
                value = int(value)
                x1, y1, x2, y2 = info['bbox']
                detected_values[char] = value
                if not conditions[char](value):
                    draw.rectangle([x1, y1, x2, y2], outline="red")
                else:
                    draw.rectangle([x1, y1, x2, y2], outline="green")

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), detected_values

# Streamlit application setup
st.title('Real-time Video Processing with OCR')
st.text('Choose between uploading a video, using your camera, or streaming via IP Webcam.')

# Option to choose between file upload, camera, or IP Webcam
option = st.selectbox("Choose input method", ["Upload Video", "Use Camera", "IP Webcam"])

if option == "Use Camera":
    st.text("Using camera for real-time video feed. Press 'q' to stop.")
    cap = cv2.VideoCapture(0)  # Use the camera (index 0 for default camera)
    frame_window = st.image([])

    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while True:
            ret, frame = cap.read()  # Capture frame-by-frame
            if not ret:
                st.warning("Camera feed not available.")
                break

            if frame_count % int(fps) == 0:
                processed_frame, detected_values = process_frame(frame)
                frame_window.image(processed_frame, channels='BGR')
                print(f"Frame {frame_count // int(fps) + 1}: {detected_values}")

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

elif option == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video file", type=["wmv", "mp4", "avi"])

    if uploaded_file is not None:
        video_path = uploaded_file.name

        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        cap = cv2.VideoCapture(video_path)

        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_window = st.image([])
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % int(fps) == 0:
                    processed_frame, detected_values = process_frame(frame)
                    frame_window.image(processed_frame, channels='BGR')
                    print(f"Frame {frame_count // int(fps) + 1}: {detected_values}")

                frame_count += 1

            cap.release()

elif option == "IP Webcam":
    st.text("Stream video via IP Webcam. Enter the stream URL from your phone (e.g., http://192.168.1.x:8080/video).")
    
    ip_webcam_url = st.text_input("Enter the IP Webcam URL:", "http://192.168.1.x:8080/video")
    
    # Initialize a session state variable to manage stream status
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False

    if st.button("Start/Stop IP Webcam Stream"):
        st.session_state.streaming = not st.session_state.streaming  # Toggle streaming status
        if st.session_state.streaming:
            st.text("Streaming IP Webcam video feed...")
            cap = cv2.VideoCapture(ip_webcam_url)

            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_window = st.image([])
                detected_values_display = st.empty()
                frame_count = 0

                while st.session_state.streaming:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("IP Webcam feed not available.")
                        break

                    if frame_count % int(fps) == 0:
                        processed_frame, detected_values = process_frame(frame)
                        frame_window.image(processed_frame, channels='BGR')
                        detected_values_display.text(f"Detected Values: {detected_values}")
                        print(f"Frame {frame_count // int(fps) + 1}: {detected_values}")

                    frame_count += 1

                cap.release()
                st.text("Stopped the IP Webcam stream.")

        else:
            st.text("Stopped the IP Webcam stream.")
