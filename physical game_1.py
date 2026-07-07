import cv2
import numpy as np
import onnxruntime as ort
import pygame
import sys
import random


pygame.init()
screen=pygame.display.set_mode((500,500))
clock = pygame.optimizations = pygame.time.Clock()
pygame.font.init()
x=250
score=0
enemies=[]
for _ in range(0,5):
    enemy=pygame.Rect(random.randrange(10,490),random.randrange(-20,-1),10,10)
    enemies.append(enemy)

my_font=pygame.font.SysFont(None,50)


# 1. Initialize ONNX Runtime with your GTX 1650 CUDA cores
providers = ['CUDAExecutionProvider']
session = ort.InferenceSession("yolo11n-pose.onnx", providers=providers)

active_provider = session.get_providers()[0]
print(f"🔥 Active Execution Backend: {session.get_provider_options()}")

# Get model input names and dimensions
input_name = session.get_inputs()[0].name
model_width, model_height = 640, 640

# Target keypoint index for Nose is 0
NOSE_INDEX = 0

cap = cv2.VideoCapture(0)
runnning=True
while runnning:

    success, frame = cap.read()
    if not success:
        break
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # User clicked the 'X' button
            runnning=False



    frame = cv2.flip(frame, 1)
    orig_h, orig_w, _ = frame.shape

    # 2. Pre-process frame to feed into YOLO ONNX layout [1, 3, 640, 640]
    input_img = cv2.resize(frame, (model_width, model_height))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img.transpose(2, 0, 1)  # Change HWC to CHW
    input_img = np.expand_dims(input_img, axis=0).astype(np.float32)
    input_img /= 255.0  # Normalize pixel values to 0.0 - 1.0

    # 3. Fast Inference Engine execution on GPU
    outputs = session.run(None, {input_name: input_img})
    
    # YOLO Pose outputs shape is typically [1, 56, 8400] (box coordinates + 17 keypoints)
    output_data = np.squeeze(outputs[0])

    # Extract predictions where confidence is high
    # Transpose to iterate over detections easily [8400, 56]
    output_data = output_data.T
    
    # Simple check for the strongest human detection candidate in the frame
    # (Columns 4 is box confidence, columns after 5 are keypoints)
    best_detection = max(output_data, key=lambda x: x[4])
    confidence = best_detection[4]

    if confidence > 0.5:
        # Extract Keypoint 0 (Nose). In YOLO pose arrays, keypoints start at index 5.
        # Structure per keypoint: [x, y, visibility]
        start_idx = 5 + (NOSE_INDEX * 3)
        
        # Coordinates out of ONNX are mapped relative to the 640x640 processing scale
        raw_x = best_detection[start_idx]
        raw_y = best_detection[start_idx + 1]
        
        # Re-scale back up to your true webcam pixel boundaries
        pixel_x = int((raw_x / model_width) * orig_w)
        pixel_y = int((raw_y / model_height) * orig_h)

        if 0 < pixel_x < orig_w and 0 < pixel_y < orig_h:
            # 4. Normalize coordinates for gaming engine stream (0.0 to 1.0)
            norm_x = pixel_x / orig_w
            norm_y = pixel_y / orig_h


            
            #y=norm_y*500



            #print(f"🎯 Nose Position -> X: {norm_x:.3f}, Y: {norm_y:.3f}")

            ## Draw targeting elements on screen
            #cv2.circle(frame, (pixel_x, pixel_y), 8, (0, 255, 0), -1)
            #cv2.circle(frame, (pixel_x, pixel_y), 16, (255, 0, 0), 2)
            #cv2.putText(
            #    frame, f"Nose X: {norm_x:.2f} Y: {norm_y:.2f}", 
            #    (pixel_x + 15, pixel_y - 15), 
            #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            #)
        #pygame logic
        x=norm_x*500
        screen.fill((0,0,0))
        player=pygame.draw.rect(screen,(255,0,0),(x,400,70,10))

        
        for enemy in enemies:
            enemy.y+=2
            if enemy.y>505:
                enemy.y=random.randrange(-20,-1)
                enemy.x=random.randrange(10,490)
                score-=1
            pygame.draw.rect(screen,(0,255,0),enemy)
        collide=player.collidelist(enemies)
        if collide !=-1:
            enemies[collide].y=random.randrange(-20,-1)
            enemies[collide].x=random.randrange(10,490)
            score+=1
        text_surface=my_font.render(f"score: {score}",True,(0,0,255),)
        screen.blit(text_surface,(350,0))




    #cv2.imshow("ONNX Runtime GPU - High Speed Nose Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
    pygame.display.flip()
    clock.tick(60)
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()