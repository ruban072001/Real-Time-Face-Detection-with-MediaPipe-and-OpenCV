**Real-Time Face Detection with MediaPipe and OpenCV 👁️📷**

This project performs real-time face detection using MediaPipe and OpenCV. The application detects faces from the webcam feed and draws bounding boxes around them with enhanced visual effects to highlight the corners. It also displays the detection confidence and calculates the frames per second (FPS) for real-time performance analysis.

**Features ✨**

**Real-time face detection:** Capture frames from your webcam and detect faces instantly.

**Bounding boxes with styled corners:** Visualize detected faces with custom bounding box corners for clarity.

**Accuracy display:** See the detection confidence score for each detected face.

**FPS display:** Real-time FPS counter to track the performance of the application.

**Technologies Used 🛠️**

**Python:** The programming language used to implement the project.
**OpenCV:** For real-time webcam capture and frame processing.
**MediaPipe:** For face detection using pre-trained models.
**Time:** For FPS calculation.

**How It Works 🔍**

1.The webcam feed is captured using OpenCV.

2.MediaPipe detects faces and returns bounding box coordinates and confidence scores.

3.Bounding boxes are drawn around the faces, with corner enhancements to make the box more visually appealing.

4.FPS is calculated to monitor the performance of the system.

**Installation ⚙️**

**Clone the repository:**
git clone https://github.com/yourusername/Face-Detection-MediaPipe-OpenCV.git

**Navigate to the project directory:**
cd Face-Detection-MediaPipe-OpenCV

**Install dependencies:**
pip install -r requirements.txt

Here's a **sample requirements.txt:**
opencv-python,
mediapipe.

**Run the project:**
python face_detection.py

**Usage 🎮**

1.Ensure your webcam is connected.

2.Run the project to see real-time face detection with accuracy and FPS displayed on the screen.

3.Press q to exit the application.

**Project Structure 📂**

Face-Detection-MediaPipe-OpenCV/

│

├── face_detection.py          # Main script to run the face detection

├── requirements.txt           # Project dependencies

└── README.md                  # Project documentation

**Future Improvements 🛠️**

1.Implement detection of multiple faces with unique IDs.

2.Add support for face recognition or emotion detection.

3.Explore additional features like face landmark tracking.
