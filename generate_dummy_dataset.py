import cv2
import os

def generate_dataset(person_name, num_images=20):
    dataset_path = "dataset"
    person_path = os.path.join(dataset_path, person_name)
    
    if not os.path.exists(person_path):
        os.makedirs(person_path)
        
    cam = cv2.VideoCapture(0)
    
    # Check if a Haar Cascade is available for face detection
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    print(f"\n[INFO] Initializing capture for '{person_name}'. Look at the camera and wait...")
    print("[INFO] We will capture {} images.".format(num_images))
    print("[INFO] Press 'q' to quit early.")
    
    count = 0
    while count < num_images:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to grab frame from camera.")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face region, resize, and save
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))
            
            img_path = os.path.join(person_path, f"{person_name}_{count}.jpg")
            cv2.imwrite(img_path, face_img)
            
            count += 1
            print(f"Captured image {count}/{num_images}")
            
            # Pause slightly between captures
            cv2.waitKey(200)
            
            if count >= num_images:
                break
                
        cv2.imshow('Capturing Dataset - Press Q to Quit', frame)
        
        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    print(f"\n[SUCCESS] Captured {count} images for '{person_name}' in {person_path}")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== Face Recognition Dataset Generator ===")
    name = input("Enter the name of the person: ").strip()
    if name:
        num = input("How many images to capture? (default 20): ").strip()
        num = int(num) if num.isdigit() else 20
        generate_dataset(name, num)
    else:
        print("[ERROR] Name cannot be empty.")
