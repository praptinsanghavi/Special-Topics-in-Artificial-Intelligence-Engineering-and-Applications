def preprocess_frame(frame):
    """Convert frame to grayscale and resize to 84x84"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized / 255.0
