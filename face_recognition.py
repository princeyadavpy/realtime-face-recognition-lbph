import cv2
import os
import numpy as np

# ------------------ SETUP ------------------
# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Dataset folder for storing faces
dataset_path = "faces_dataset"
os.makedirs(dataset_path, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("\n[INFO] Controls:")
print("Press 'c' → Capture new face samples")
print("Press 'v' → Verify a person")
print("Press 'q' → Quit\n")

# ------------------ FUNCTIONS ------------------
def capture_faces(name, samples=30):
    """Capture face samples for a given person name and display progress."""
    person_folder = os.path.join(dataset_path, name)
    os.makedirs(person_folder, exist_ok=True)

    count = 0
    print(f"[INFO] Capturing faces for {name}... Look at the camera.")
    while count < samples:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            file_path = os.path.join(person_folder, f"{count}.jpg")
            cv2.imwrite(file_path, face)
            count += 1

            # Draw rectangle and progress text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing: {name} ({count}/{samples})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"[INFO] Captured {count} face samples for {name}")
    cv2.destroyWindow("Capturing Faces")
    return count


def train_recognizer():
    """Train the recognizer with all saved face samples."""
    faces, labels = [], []
    label_map = {}
    label_id = 0

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        label_map[label_id] = person_name
        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(label_id)
        label_id += 1

    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        print("[INFO] Training completed successfully.")
    else:
        print("[WARNING] No faces found to train on.")
    return label_map


def verify_person(name_to_verify):
    """Verify if the person in front of the camera matches the given name."""
    if not label_map:
        print("[ERROR] Train the model first by adding faces.")
        return

    print(f"[INFO] Verification started for '{name_to_verify}'. Press 'q' to exit verification.")
    verified = False
    attempts = 0
    while attempts < 50:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face)
            predicted_name = label_map.get(label, "Unknown")

            if predicted_name == name_to_verify and confidence < 70:
                cv2.putText(frame, "Verified ✅", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                verified = True
            else:
                cv2.putText(frame, "Not Verified ❌", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Verification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        attempts += 1

    cv2.destroyWindow("Verification")
    if verified:
        print(f"[RESULT] ✅ {name_to_verify} is Verified")
    else:
        print(f"[RESULT] ❌ {name_to_verify} Not Verified")


# ------------------ MAIN LOOP ------------------
label_map = train_recognizer()  # Train at startup (if faces exist)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        if len(label_map) > 0:
            label, confidence = recognizer.predict(face)
            name = label_map.get(label, "Unknown")
            text = f"{name} ({int(confidence)})"
        else:
            text = "Face Detected"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Detection & Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        person_name = input("Enter name of person: ").strip()
        if person_name:
            count = capture_faces(person_name)
            if count > 0:
                label_map = train_recognizer()  # Auto-train after capturing
    elif key == ord('v'):
        person_name = input("Enter name to verify: ").strip()
        if person_name:
            verify_person(person_name)

cap.release()
cv2.destroyAllWindows()
