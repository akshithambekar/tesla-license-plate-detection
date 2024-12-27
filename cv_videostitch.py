import cv2
import numpy as np

def undistort_fisheye_frame(frame, dimension):
    height, width = frame.shape[:2]
    K = np.array([[dimension, 0, width / 2],
                  [0, dimension, height / 2],
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([0, 0, 0, 0], dtype=np.float32)

    # Perform fisheye undistortion
    undistorted_frame = cv2.fisheye.undistortImage(frame, K, D, Knew=K)

    return undistorted_frame

def resize_frame(frame, target_height): # resize frame to target height while maintaining aspect ratio
    ratio = target_height / frame.shape[0]
    dim = (int(frame.shape[1] * ratio), target_height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized_frame

def stitch_frames(frame1, frame2, frame3, frame4):
    overlap_width = 100 
    crop_frame1 = frame1[:, :-overlap_width, :]
    crop_frame2 = frame2[:, overlap_width:, :]
    crop_frame3 = frame3[:, overlap_width:, :]
    target_height = max(crop_frame1.shape[0], crop_frame2.shape[0], crop_frame3.shape[0], frame4.shape[0])
    resized_frame1 = resize_frame(crop_frame1, target_height)
    resized_frame2 = resize_frame(crop_frame2, target_height)
    resized_frame3 = resize_frame(crop_frame3, target_height)
    resized_frame4 = resize_frame(frame4, target_height)
    stitched_frame = np.hstack((resized_frame4, resized_frame1, resized_frame2, resized_frame3))
    return stitched_frame

if __name__ == "__main__":
    video_paths = ['./TeslaCam Footage/front.mp4', './TeslaCam Footage/right_repeater.mp4', './TeslaCam Footage/rear.mp4', './TeslaCam Footage/left_repeater.mp4']
    distortion_strength = 0.2
    cap1 = cv2.VideoCapture(video_paths[0])
    cap2 = cv2.VideoCapture(video_paths[1])
    cap3 = cv2.VideoCapture(video_paths[2])
    cap4 = cv2.VideoCapture(video_paths[3])
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('combined_video.mp4', fourcc, 20.0, (width * 4, height))

    while all([cap.isOpened() for cap in [cap1, cap2, cap3, cap4]]):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()
        if not (ret1 and ret2 and ret3 and ret4):
            break
        undistorted_frame1 = undistort_fisheye_frame(frame1, 800)
        undistorted_frame2 = undistort_fisheye_frame(frame2, 800)
        undistorted_frame3 = undistort_fisheye_frame(frame3, 600)
        undistorted_frame4 = undistort_fisheye_frame(frame4, 800)
        combined_frame = stitch_frames(undistorted_frame2, undistorted_frame3, undistorted_frame4, undistorted_frame1)
        out.write(combined_frame)
        cv2.imshow('Combined Video', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    for cap in [cap1, cap2, cap3, cap4]:
        cap.release()
    out.release()
    cv2.destroyAllWindows()