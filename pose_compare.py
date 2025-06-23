import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Video paths
coach_path = r"C:\Users\likit\Desktop\pose_comparisons\coach.mp4"
student_path = r"C:\Users\likit\Desktop\pose_comparisons\student.mp4"
output_path = r"C:\Users\likit\Desktop\pose_comparisons\pose_comparison_output.mp4"

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw = mp.solutions.drawing_utils

# Video capture
cap1 = cv2.VideoCapture(coach_path)
cap2 = cv2.VideoCapture(student_path)

fps = int(cap1.get(cv2.CAP_PROP_FPS))
w = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Note: height because of 90-degree rotation
h = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 2, h))

def get_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    return result.pose_landmarks

def compare_landmarks(lm1, lm2):
    if not lm1 or not lm2:
        return 0.0
    l1 = np.array([[p.x, p.y, p.z] for p in lm1.landmark])
    l2 = np.array([[p.x, p.y, p.z] for p in lm2.landmark])
    l1 -= l1[0]
    l2 -= l2[0]
    return np.mean(np.linalg.norm(l1 - l2, axis=1))

# Process frames
diffs = []
while cap1.isOpened() and cap2.isOpened():
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2:
        break

    # Fix rotation (rotate 90° clockwise)
    f1 = cv2.rotate(f1, cv2.ROTATE_90_CLOCKWISE)
    f2 = cv2.rotate(f2, cv2.ROTATE_90_CLOCKWISE)

    lm1 = get_landmarks(f1)
    lm2 = get_landmarks(f2)

    if lm1: draw.draw_landmarks(f1, lm1, mp_pose.POSE_CONNECTIONS)
    if lm2: draw.draw_landmarks(f2, lm2, mp_pose.POSE_CONNECTIONS)

    diff = compare_landmarks(lm1, lm2)
    diffs.append(diff)
    similarity = max(0, 100 - diff * 100)
    text = f"Similarity: {similarity:.2f}%"

    for frame in [f1, f2]:
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    combined = np.hstack((f1, f2))
    out.write(combined)

# Cleanup
cap1.release()
cap2.release()
out.release()
pose.close()

if diffs:
    avg_sim = max(0, 100 - np.mean(diffs) * 100)
    print(f"✅ Average Similarity: {avg_sim:.2f}%")
    print(f"🎯 Output saved to: {output_path}")
else:
    print("⚠️ Pose not detected in videos.")
