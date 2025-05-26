import cv2
import os

def video_to_frames_interval(video_path, output_folder, interval_seconds=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            print(f"保存第 {saved_count} 帧: {frame_filename}")

        frame_count += 1

    cap.release()
    print(f"完成，共保存 {saved_count} 帧到文件夹 {output_folder}")

if __name__ == "__main__":
    video_path = "mp4/Aoteman.mp4"
    output_folder = "output_frames"
    video_to_frames_interval(video_path, output_folder, interval_seconds=0.5)
