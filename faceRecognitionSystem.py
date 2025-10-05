import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import pickle
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import json
import shutil

class FaceRecognitionSystem:
    def __init__(self, threshold=0.6, db_path="face_database.pkl", 
                 unauthorized_dir="unauthorized_captures"):
        """
        Initialize the face recognition system
        
        Args:
            threshold: Cosine similarity threshold for access (0.6 = 60% similarity)
            db_path: Path to save/load authorized faces database
            unauthorized_dir: Directory to save unauthorized person images
        """
        print("Initializing Face Recognition System...")
        
        # Load OpenCV's DNN face detector
        print("Loading face detector...")
        self.face_detector = cv2.dnn.readNetFromCaffe(
            self.download_face_detector_prototxt(),
            self.download_face_detector_model()
        )
        
        # Download and load ArcFace model from Hugging Face
        print("Downloading ArcFace model from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="garavv/arcface-onnx",
            filename="arc.onnx"
        )
        self.arcface_session = ort.InferenceSession(model_path)
        
        self.threshold = threshold
        self.db_path = db_path
        self.unauthorized_dir = unauthorized_dir
        self.authorized_faces = {}
        self.pending_approvals = {}  # Store pending unauthorized faces
        
        # Create directory for unauthorized captures
        os.makedirs(unauthorized_dir, exist_ok=True)
        
        # Load existing databases
        self.load_database()
        self.load_pending_approvals()
        
        print(f"System initialized. Threshold: {threshold}")
        print(f"Authorized faces in database: {len(self.authorized_faces)}")
        print(f"Pending approvals: {len(self.pending_approvals)}")
    
    def download_face_detector_prototxt(self):
        """Download face detector prototxt file"""
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        prototxt_path = "deploy.prototxt"
        
        if not os.path.exists(prototxt_path):
            print("Downloading face detector prototxt...")
            import urllib.request
            urllib.request.urlretrieve(url, prototxt_path)
        
        return prototxt_path
    
    def download_face_detector_model(self):
        """Download face detector model file"""
        url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        
        if not os.path.exists(model_path):
            print("Downloading face detector model (this may take a moment)...")
            import urllib.request
            urllib.request.urlretrieve(url, model_path)
        
        return model_path
    
    def detect_faces(self, frame, confidence_threshold=0.5):
        """
        Detect faces in frame using OpenCV DNN
        
        Args:
            frame: Input image (BGR format)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of face bounding boxes [(x1, y1, x2, y2), ...]
        """
        h, w = frame.shape[:2]
        
        # Prepare image for face detector
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype(int)
                
                # Ensure box is within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Only add if box is valid
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2, y2))
        
        return faces
    
    def align_face(self, frame, bbox, target_size=(112, 112)):
        """
        Extract and align face from bounding box
        
        Args:
            frame: Input image (BGR format)
            bbox: Face bounding box (x1, y1, x2, y2)
            target_size: Output size for face
            
        Returns:
            Aligned face image (RGB format)
        """
        x1, y1, x2, y2 = bbox
        
        # Extract face region with some padding
        face_img = frame[y1:y2, x1:x2]
        
        # Convert to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        face_img = cv2.resize(face_img, target_size)
        
        return face_img
    
    def preprocess_face(self, face_img):
        """Preprocess face image for ArcFace model"""
        # Ensure size is 112x112
        if face_img.shape[:2] != (112, 112):
            face_img = cv2.resize(face_img, (112, 112))
        
        # Normalize to [-1, 1]
        face_img = (face_img.astype(np.float32) - 127.5) / 128.0
        
        # Add batch dimension (NHWC format)
        face_img = np.expand_dims(face_img, axis=0)
        
        return face_img
    
    def get_embedding(self, face_img):
        """Generate embedding using ArcFace model"""
        preprocessed = self.preprocess_face(face_img)
        
        # Get input name
        input_name = self.arcface_session.get_inputs()[0].name
        
        # Run inference
        embedding = self.arcface_session.run(None, {input_name: preprocessed})[0]
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.flatten()
    
    def add_authorized_face(self, name, face_img):
        """
        Add a new authorized face to the database
        
        Args:
            name: Person's name/ID
            face_img: Aligned face image (RGB)
        """
        embedding = self.get_embedding(face_img)
        self.authorized_faces[name] = embedding
        self.save_database()
        print(f"Added {name} to authorized faces database")
    
    def save_database(self):
        """Save authorized faces database to disk"""
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.authorized_faces, f)
        print(f"Database saved to {self.db_path}")
    
    def load_database(self):
        """Load authorized faces database from disk"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                self.authorized_faces = pickle.load(f)
            print(f"Loaded {len(self.authorized_faces)} authorized faces from database")
        else:
            print("No existing database found. Starting fresh.")
    
    def save_pending_approvals(self):
        """Save pending approvals to disk"""
        pending_file = os.path.join(self.unauthorized_dir, "pending_approvals.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_pending = {}
        for capture_id, data in self.pending_approvals.items():
            serializable_pending[capture_id] = {
                'timestamp': data['timestamp'],
                'image_path': data['image_path'],
                'full_frame_path': data['full_frame_path'],
                'similarity_score': float(data['similarity_score']),
                'best_match': data['best_match']
            }
        
        with open(pending_file, 'w') as f:
            json.dump(serializable_pending, f, indent=2)
    
    def load_pending_approvals(self):
        """Load pending approvals from disk"""
        pending_file = os.path.join(self.unauthorized_dir, "pending_approvals.json")
        if os.path.exists(pending_file):
            with open(pending_file, 'r') as f:
                self.pending_approvals = json.load(f)
            print(f"Loaded {len(self.pending_approvals)} pending approvals")
        else:
            print("No pending approvals found.")
    
    def capture_unauthorized_person(self, frame, face_img, similarity_score, best_match):
        """
        Capture and save an unauthorized person's image
        
        Args:
            frame: Full frame from camera
            face_img: Cropped face image (RGB)
            similarity_score: Best similarity score achieved
            best_match: Name of best match (if any)
            
        Returns:
            capture_id: Unique ID for this capture
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        capture_id = f"unauthorized_{timestamp}"
        
        # Save full frame
        frame_path = os.path.join(self.unauthorized_dir, f"{capture_id}_full.jpg")
        cv2.imwrite(frame_path, frame)
        
        # Save cropped face
        face_path = os.path.join(self.unauthorized_dir, f"{capture_id}_face.jpg")
        face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(face_path, face_bgr)
        
        # Store in pending approvals
        self.pending_approvals[capture_id] = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_path': face_path,
            'full_frame_path': frame_path,
            'similarity_score': similarity_score,
            'best_match': best_match
        }
        
        self.save_pending_approvals()
        
        print(f"\n[ALERT] Unauthorized person captured! ID: {capture_id}")
        print(f"        Saved to: {frame_path}")
        print(f"        Similarity to {best_match}: {similarity_score:.2%}" if best_match else "")
        
        return capture_id
    
    def recognize_face(self, face_img):
        """
        Recognize a face against authorized database
        
        Args:
            face_img: Aligned face image (RGB)
            
        Returns:
            tuple: (name, similarity_score, access_granted)
        """
        if len(self.authorized_faces) == 0:
            return None, 0.0, False
        
        # Get embedding for query face
        query_embedding = self.get_embedding(face_img)
        
        # Compare with all authorized faces
        best_match = None
        best_similarity = -1
        
        for name, auth_embedding in self.authorized_faces.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                auth_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        access_granted = best_similarity >= self.threshold
        
        return best_match, best_similarity, access_granted
    
    def register_face_from_webcam(self, name):
        """
        Register a new authorized face using webcam
        
        Args:
            name: Person's name/ID
        """
        cap = cv2.VideoCapture(0)
        print(f"\nRegistering face for: {name}")
        print("Position your face in front of the camera. Press SPACE to capture, ESC to cancel.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Draw rectangles around detected faces
            display_frame = frame.copy()
            for bbox in faces:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, "Face Detected", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.putText(display_frame, "Press SPACE to capture, ESC to cancel", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Register Face', display_frame)
            
            key = cv2.waitKey(1)
            
            # ESC to cancel
            if key == 27:
                print("Registration cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            # SPACE to capture
            if key == 32 and len(faces) > 0:
                # Use the first detected face
                bbox = faces[0]
                
                # Extract and align face
                face_img = self.align_face(frame, bbox)
                
                # Add to database
                self.add_authorized_face(name, face_img)
                
                print(f"Successfully registered {name}!")
                cap.release()
                cv2.destroyAllWindows()
                return True
        
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    def run_access_control(self):
        """
        Run real-time access control system with unauthorized capture
        """
        cap = cv2.VideoCapture(0)
        print("\n=== Access Control System Active ===")
        print("Press 'q' to quit, 'a' to open admin panel")
        
        # Track recently captured unauthorized persons to avoid spam
        recent_captures = {}
        capture_cooldown = 5.0  # seconds between captures of same person
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            display_frame = frame.copy()
            
            # If no face detected, skip recognition
            if len(faces) == 0:
                cv2.putText(display_frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                # Process each detected face
                for bbox in faces:
                    x1, y1, x2, y2 = bbox
                    
                    # Extract and align face
                    face_img = self.align_face(frame, bbox)
                    
                    # Recognize face
                    name, similarity, access_granted = self.recognize_face(face_img)
                    
                    # Draw results
                    if access_granted:
                        color = (0, 255, 0)  # Green for granted
                        status = "ACCESS GRANTED"
                        label = f"{name} ({similarity:.2%})"
                    else:
                        color = (0, 0, 255)  # Red for denied
                        status = "ACCESS DENIED"
                        if name:
                            label = f"Best match: {name} ({similarity:.2%})"
                        else:
                            label = "No authorized faces in database"
                        
                        # Capture unauthorized person (with cooldown)
                        current_time = datetime.now().timestamp()
                        face_key = f"{x1}_{y1}"  # Simple position-based key
                        
                        should_capture = True
                        if face_key in recent_captures:
                            time_elapsed = current_time - recent_captures[face_key]
                            if time_elapsed < capture_cooldown:
                                should_capture = False
                        
                        if should_capture:
                            capture_id = self.capture_unauthorized_person(
                                frame, face_img, similarity, name
                            )
                            recent_captures[face_key] = current_time
                            
                            # Show alert on frame
                            cv2.putText(display_frame, "CAPTURED FOR ADMIN REVIEW", 
                                      (x1, y2+20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw status
                    cv2.putText(display_frame, status, (x1, y1-30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Draw match info
                    cv2.putText(display_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show pending approvals count
            if len(self.pending_approvals) > 0:
                cv2.putText(display_frame, 
                          f"Pending approvals: {len(self.pending_approvals)} (Press 'a')", 
                          (10, display_frame.shape[0] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Display frame
            cv2.imshow('Face Recognition Access Control', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Quit on 'q'
            if key == ord('q'):
                break
            
            # Open admin panel on 'a'
            if key == ord('a'):
                cap.release()
                cv2.destroyAllWindows()
                self.admin_panel()
                cap = cv2.VideoCapture(0)
        
        cap.release()
        cv2.destroyAllWindows()
        print("Access control system stopped")
    
    def admin_panel(self):
        """
        Admin panel to review and approve/reject unauthorized persons
        """
        while True:
            print("\n" + "="*60)
            print("ADMIN PANEL - Unauthorized Person Review")
            print("="*60)
            
            if len(self.pending_approvals) == 0:
                print("No pending approvals.")
                input("\nPress Enter to return to main menu...")
                return
            
            # Display pending approvals
            pending_list = list(self.pending_approvals.items())
            for idx, (capture_id, data) in enumerate(pending_list, 1):
                print(f"\n{idx}. Capture ID: {capture_id}")
                print(f"   Timestamp: {data['timestamp']}")
                print(f"   Best Match: {data['best_match']} ({data['similarity_score']:.2%})" 
                      if data['best_match'] else "   No match in database")
                print(f"   Image: {data['image_path']}")
            
            print("\n" + "="*60)
            print("Options:")
            print("  [number] - Review specific capture")
            print("  [b] - Back to main menu")
            print("="*60)
            
            choice = input("Enter choice: ").strip().lower()
            
            if choice == 'b':
                return
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(pending_list):
                    capture_id, data = pending_list[idx]
                    self.review_capture(capture_id, data)
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input.")
    
    def review_capture(self, capture_id, data):
        """
        Review a specific unauthorized capture
        
        Args:
            capture_id: Unique capture ID
            data: Capture data dictionary
        """
        print("\n" + "="*60)
        print(f"Reviewing: {capture_id}")
        print("="*60)
        print(f"Timestamp: {data['timestamp']}")
        print(f"Best Match: {data['best_match']} ({data['similarity_score']:.2%})" 
              if data['best_match'] else "No match in database")
        
        # Display images
        face_img = cv2.imread(data['image_path'])
        full_frame = cv2.imread(data['full_frame_path'])
        
        if face_img is not None:
            cv2.imshow('Face Capture', face_img)
        if full_frame is not None:
            cv2.imshow('Full Frame', full_frame)
        
        print("\nOptions:")
        print("  [a] - Approve and add to authorized users")
        print("  [r] - Reject and delete")
        print("  [k] - Keep for later review")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('a'):
                cv2.destroyAllWindows()
                name = input("\nEnter name for this person: ").strip()
                if name:
                    # Load the face image and add to authorized
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    self.add_authorized_face(name, face_rgb)
                    
                    # Remove from pending
                    del self.pending_approvals[capture_id]
                    self.save_pending_approvals()
                    
                    print(f"✓ {name} added to authorized users!")
                else:
                    print("Invalid name. Keeping in pending.")
                break
            
            elif key == ord('r'):
                cv2.destroyAllWindows()
                confirm = input("\nAre you sure you want to reject and delete? (y/n): ").strip().lower()
                if confirm == 'y':
                    # Delete images
                    try:
                        os.remove(data['image_path'])
                        os.remove(data['full_frame_path'])
                    except:
                        pass
                    
                    # Remove from pending
                    del self.pending_approvals[capture_id]
                    self.save_pending_approvals()
                    
                    print("✓ Capture rejected and deleted.")
                break
            
            elif key == ord('k'):
                cv2.destroyAllWindows()
                print("Kept for later review.")
                break
        
        input("\nPress Enter to continue...")


def main():
    """Main function with menu interface"""
    system = FaceRecognitionSystem(threshold=0.6)
    
    while True:
        print("\n" + "="*60)
        print("Face Recognition Access Control System")
        print("="*60)
        print("1. Register new authorized face")
        print("2. Run access control system")
        print("3. Admin panel - Review unauthorized captures")
        print("4. List authorized faces")
        print("5. Change similarity threshold")
        print("6. Clear all pending approvals")
        print("7. Exit")
        print("="*60)
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            name = input("Enter person's name/ID: ").strip()
            if name:
                system.register_face_from_webcam(name)
            else:
                print("Invalid name. Please try again.")
        
        elif choice == '2':
            if len(system.authorized_faces) == 0:
                print("\nWarning: No authorized faces in database!")
                print("Please register at least one face first.")
            else:
                system.run_access_control()
        
        elif choice == '3':
            system.admin_panel()
        
        elif choice == '4':
            if len(system.authorized_faces) == 0:
                print("\nNo authorized faces in database.")
            else:
                print(f"\nAuthorized Faces ({len(system.authorized_faces)}):")
                for i, name in enumerate(system.authorized_faces.keys(), 1):
                    print(f"  {i}. {name}")
        
        elif choice == '5':
            try:
                new_threshold = float(input(f"Current threshold: {system.threshold}\nEnter new threshold (0.0-1.0): "))
                if 0.0 <= new_threshold <= 1.0:
                    system.threshold = new_threshold
                    print(f"Threshold updated to {new_threshold}")
                else:
                    print("Threshold must be between 0.0 and 1.0")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        elif choice == '6':
            confirm = input(f"Clear all {len(system.pending_approvals)} pending approvals? (y/n): ").strip().lower()
            if confirm == 'y':
                # Delete all images
                for capture_id, data in system.pending_approvals.items():
                    try:
                        os.remove(data['image_path'])
                        os.remove(data['full_frame_path'])
                    except:
                        pass
                
                system.pending_approvals = {}
                system.save_pending_approvals()
                print("✓ All pending approvals cleared.")
        
        elif choice == '7':
            print("Exiting system. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()