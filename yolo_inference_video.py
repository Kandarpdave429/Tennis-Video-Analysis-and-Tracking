from ultralytics import YOLO

model=YOLO('yolov8x')

# Inference
result= model.predict(r'D:\Tennis_Analysis\input_videos\input_video.mp4', save=True)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)