import os
import sys
import cv2
import argparse
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from ai.face_recognition import SCRFDDetector, ArcFaceEmbedder, align_face

# ===== ตั้งค่าสำหรับลงทะเบียนใบหน้าด้วยรูปภาพ (Image-Path Registration Config) =====
# หากต้องการลงทะเบียนใบหน้า ให้ใส่พาธรูปภาพและชื่อของคุณตรงนี้ (ระบบจะทำการประมวลผลให้เมื่อรันสคริปต์)
REGISTER_IMAGE_PATH = r"C:\Users\M S I\Desktop\project_main\ai_surveillance\Prime_image.jpg"  # ใส่พาธรูปภาพของคุณตรงนี้ เช่น "C:/Users/Name/Downloads/my_face.jpg"
REGISTER_NAME = "Prime"        # ใส่ชื่อของคุณตรงนี้ เช่น "somchai"
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Real-time Face Recognition & Registration Demo")
    parser.add_argument("--video", type=str, default="0", help="Video file path or webcam index (default: 0 for webcam)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Cosine similarity threshold for matching (default: 0.55)")
    args = parser.parse_args()

    known_faces_dir = os.path.join(project_root, "data", "known_faces")
    os.makedirs(known_faces_dir, exist_ok=True)

    print("Initializing SCRFD Face Detector and ArcFace Embedder...")
    detector = SCRFDDetector()
    embedder = ArcFaceEmbedder()
    print("AI Models loaded successfully!")

    # 1. Mode: Register Face via Image Path (Auto run if config is set)
    if REGISTER_IMAGE_PATH.strip() and REGISTER_NAME.strip():
        img_path = REGISTER_IMAGE_PATH.strip()
        name = REGISTER_NAME.strip()
        print(f"\n--- AUTO REGISTERING: '{name}' from image '{img_path}' ---")
        
        if not os.path.exists(img_path):
            print(f"Error: The image file was not found at path: {img_path}")
            return
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image at path: {img_path}")
            return
            
        faces = detector.detect(img)
        if len(faces) == 0:
            print("Error: Could not detect any faces in the provided photo. Please choose another photo.")
            return
        elif len(faces) > 1:
            print("Error: Multiple faces detected in the provided photo. Please make sure only one person is in the image.")
            return
            
        # Align face and save it to database
        aligned = align_face(img, faces[0]['kps'])
        out_path = os.path.join(known_faces_dir, f"{name}.png")
        cv2.imwrite(out_path, aligned)
        print(f"[SUCCESS] Registered and saved aligned face of '{name}' to database at: {out_path}")
        print("Tip: You can now clear the REGISTER_IMAGE_PATH and REGISTER_NAME variables to skip this step next time.\n")

    # 2. Mode: Real-time Face Recognition
    # Load and process known faces
    print("\nLoading registered faces from database...")
    known_embeddings = {}
    
    valid_extensions = (".png", ".jpg", ".jpeg")
    for file in os.listdir(known_faces_dir):
        if file.lower().endswith(valid_extensions):
            name = os.path.splitext(file)[0]
            img_path = os.path.join(known_faces_dir, file)
            img = cv2.imread(img_path)
            
            if img is not None:
                # If image is already an aligned 112x112 face crop, bypass detector and get embedding directly
                if img.shape[0] == 112 and img.shape[1] == 112:
                    emb = embedder.get_embedding(img)
                    known_embeddings[name] = emb
                    print(f"  Loaded: {name} (Pre-aligned face crop)")
                else:
                    # Otherwise, run detector and align it
                    faces = detector.detect(img)
                    if len(faces) == 1:
                        aligned = align_face(img, faces[0]['kps'])
                        emb = embedder.get_embedding(aligned)
                        known_embeddings[name] = emb
                        # Overwrite original image with the aligned version for next time
                        cv2.imwrite(img_path, aligned)
                        print(f"  Loaded: {name} (Face detected, aligned, and cached)")
                    elif len(faces) == 0:
                        print(f"  Warning: No face detected in {file}. Skipping this person.")
                    else:
                        print(f"  Warning: Multiple faces found in {file}. Skipping this person.")

    if not known_embeddings:
        print("\nWarning: No registered faces found in data/known_faces/.")
        print("You can register faces by placing photos there or running this script with --register <name>.")
    else:
        print(f"\nDatabase loaded. Total registered identities: {len(known_embeddings)}")

    # Try resolving camera index vs video path
    video_source = args.video
    if video_source.isdigit():
        video_source = int(video_source)
        
    print(f"\nOpening video stream: {args.video}...")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.video}")
        return

    print("Real-time recognition loop active. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or frame read failed.")
            break
            
        faces = detector.detect(frame)
        
        for face in faces:
            bbox = face['bbox']
            kps = face['kps']
            score = face['score']
            
            # 1. Align face
            aligned = align_face(frame, kps)
            
            # 2. Get embedding
            emb = embedder.get_embedding(aligned)
            
            # 3. Match against known database
            identity = "Stranger"
            max_sim = 0.0
            
            for name, known_emb in known_embeddings.items():
                # Dot product on L2 normalized embeddings computes Cosine Similarity directly
                sim = float(np.dot(emb, known_emb))
                if sim > max_sim:
                    max_sim = sim
                    
            if max_sim >= args.threshold:
                identity = f"{identity_clean(name)} ({max_sim:.2f})"
                box_color = (0, 255, 0) # Green for matched
            else:
                identity = f"Stranger ({max_sim:.2f})" if max_sim > 0 else "Stranger"
                box_color = (0, 0, 255) # Red for stranger
                
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)
            
            # Draw 5 facial landmarks
            for kp in kps:
                cv2.circle(frame, (kp[0], kp[1]), 3, (255, 0, 0), -1)
                
            # Display name & score overlay
            label_y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 20
            cv2.putText(frame, identity, (bbox[0], label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Show display window
        cv2.imshow("AI-Based Smart Surveillance - Face Recognition Demo", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Demo loop terminated.")

def identity_clean(name):
    # Helper to remove suffixes or clean filename patterns
    return name.replace("_", " ").title()

if __name__ == "__main__":
    main()
