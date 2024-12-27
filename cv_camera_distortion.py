import cv2
import numpy as np
from matplotlib import pyplot as plt

def undistort_fisheye_frame(frame):
    height, width = frame.shape[:2]
    K = np.array([[800, 0, width / 2], [0, 800, height / 2], [0, 0, 1]], dtype=np.float32)
    D = np.array([0, 0, 0, 0], dtype=np.float32)
    undistorted_frame = cv2.fisheye.undistortImage(frame, K, D, Knew=K)
    return undistorted_frame

if __name__ == "__main__":
    video_path = './TeslaCam Footage/rear.mp4'
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('undistorted_video.avi', fourcc, 20.0, (width * 2, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        undistorted_frame = undistort_fisheye_frame(frame)
        display_frame = np.hstack((frame, undistorted_frame))
        out.write(display_frame)
        cv2.imshow('Original vs Undistorted', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()