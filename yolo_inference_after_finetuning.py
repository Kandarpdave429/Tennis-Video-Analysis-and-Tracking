from ultralytics import YOLO

model=YOLO(r'D:\Tennis_Analysis\models\yolov5_last.pt')

# Inference
result= model.predict(r'D:\Tennis_Analysis\input_videos\input_video.mp4', conf=0.2, save=True)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)