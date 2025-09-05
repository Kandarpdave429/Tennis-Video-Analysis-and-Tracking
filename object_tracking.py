# from ultralytics import YOLO

# model=YOLO('yolov8x')

# # Inference
# result= model.track(r'D:\Tennis_Analysis\input_videos\input_video.mp4', save=True)
# # print(result)
# # print("boxes:")
# # for box in result[0].boxes:
# #     print(box)

from ultralytics import YOLO

model = YOLO('yolov8x.pt')

# Run tracking inference
results = model.track(r'D:\Tennis_Analysis\input_videos\input_video.mp4', save=True)

# Collect frame-wise detection stats
frame_stats = []
for i, frame_result in enumerate(results[0].boxes):
    frame_stats.append(len(frame_result))

avg_detections = sum(frame_stats) / len(frame_stats)
print(f"Average detections per frame: {avg_detections}")
print(f"Total frames analyzed: {len(frame_stats)}")
