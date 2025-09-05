from roboflow import Roboflow
rf = Roboflow(api_key="Your API Key")
project = rf.workspace("viren-dhanwani").project("tennis-ball-detection")
version = project.version(6)
dataset = version.download("yolov5")                