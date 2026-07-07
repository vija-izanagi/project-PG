better yolo model means better accuracy but low fps /jitter
name of model : corresponding python name
Nano: "yolo11n-pose"
Small: "yolo11s-pose"
Medium: "yolo11m-pose"
Large: "yolo11l-pose"
Extra Large: "yolo11x-pose"


if you want to change model 
step1:download for_downloading_different_yollo_model.py 
step2:go to line for look for model = YOLO("yolo11n-pose.pt") cgange this to your model name in this format model = YOLO("your model name.pt")
step3:run this 
step4:open physical game_1.py and look for session = ort.InferenceSession("yolo11n-pose.onnx", providers=providers) at line 25 and change it to session = ort.InferenceSession("your model name.onnx", providers=providers)


currently it uses cpu 
