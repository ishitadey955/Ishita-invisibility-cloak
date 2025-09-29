# %%
import cv2
import numpy as np
import time

def get_background(cap, num_frames=60, flip=True):
    """
    Captures the background by averaging a number of frames.
    """
    print("Capturing background... Please stand still.")
    
    
    time.sleep(2)
    
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            return None
        if flip:
            frame = np.flip(frame, axis=1)
        frames.append(frame)
    
    
    background = np.median(frames, axis=0).astype(np.uint8)
    print("Background captured.")
    return background

def create_mask(hsv_frame):
    """
    Creates a combined mask for the BLUE color.
    """
    
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

   
    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    
   
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    return mask

def main():
    print("Starting Invisible Cloak (Blue Detection)... Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

   
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 520)

   
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Webcam resolution set to: {actual_width}x{actual_height}")

    background = get_background(cap)
    if background is None:
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        
        frame = np.flip(frame, axis=1)

       
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
        blue_mask = create_mask(hsv)
        
       
        blue_mask_inv = cv2.bitwise_not(blue_mask)

     
        part1 = cv2.bitwise_and(background, background, mask=blue_mask)

        
        part2 = cv2.bitwise_and(frame, frame, mask=blue_mask_inv)

      
        final_output = cv2.add(part1, part2)

        cv2.imshow("Invisible Cloak (Blue)", final_output)

       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

   
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# %%


# %%


# %%



