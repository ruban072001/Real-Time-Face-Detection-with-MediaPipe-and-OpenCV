import cv2
import mediapipe as mp
import time

class FaceDetection:
    
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        # Initialize MediaPipe Face Detection with the passed parameters
        self.mpface = mp.solutions.face_detection.FaceDetection(
            self.min_detection_confidence, self.model_selection
        )
        self.mpdraw = mp.solutions.drawing_utils

    def draw_bounding_box(self, frame, x, y, width, height, w, h):
        # Draw the full bounding box with corners
        # Bounding box
        cv2.rectangle(frame, (int(x * w), int(y * h)), 
                      (int((x + width) * w), int((y + height) * h)), 
                      (0, 200, 0), 1)

        # Draw thicker corners
        # Top left
        cv2.line(frame, (int(x * w), int(y * h)), 
                      (int(x * w), int(y * h) + 10), 
                      (0, 0, 255), 4)
        cv2.line(frame, (int(x * w), int(y * h)), 
                      (int(x * w) + 10, int(y * h)), 
                      (0, 0, 255), 4)
        # Top right
        cv2.line(frame, (int((x + width) * w), int(y * h)), 
                      (int((x + width) * w), int(y * h) + 10), 
                      (0, 0, 255), 4)
        cv2.line(frame, (int((x + width) * w), int(y * h)), 
                      (int((x + width) * w) - 10, int(y * h)), 
                      (0, 0, 255), 4)
        # Bottom left
        cv2.line(frame, (int(x * w), int((y + height) * h)), 
                      (int(x * w), int((y + height) * h) - 10), 
                      (0, 0, 255), 4)
        cv2.line(frame, (int(x * w), int((y + height) * h)), 
                      (int(x * w) + 10, int((y + height) * h)), 
                      (0, 0, 255), 4)
        # Bottom right
        cv2.line(frame, (int((x + width) * w), int((y + height) * h)), 
                      (int((x + width) * w), int((y + height) * h) - 10), 
                      (0, 0, 255), 4)
        cv2.line(frame, (int((x + width) * w), int((y + height) * h)), 
                      (int((x + width) * w) - 10, int((y + height) * h)), 
                      (0, 0, 255), 4)

    def detect_faces(self, frame, draw=True):
        # Convert the frame to RGB, as required by MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mpface.process(rgb)

        bboxes = []
        if results.detections:
            h, w, _ = frame.shape
            for i, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box
                x, y, width, height = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                bboxes.append([i, (x, y, width, height), detection.score[0]])

                # Draw the bounding box and accuracy score on the frame if requested
                if draw:
                    self.draw_bounding_box(frame, x, y, width, height, w, h)
                    cv2.putText(frame, f'Accuracy: {int(detection.score[0] * 100)}%', 
                                (int(x * w) - 5, int(y * h) - 10), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        return frame, bboxes


def main():
    vid = cv2.VideoCapture(0)

    # Variables for FPS calculation
    prev_time = 0
    
    # Create a FaceDetection object
    face_detector = FaceDetection(min_detection_confidence=0.5, model_selection=0)
    
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # Detect faces and draw bounding boxes
        frame, bboxes = face_detector.detect_faces(frame, draw=True)
        # if len(bboxes) > 0:
        #     print(bboxes)
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        # Show the detection results
        cv2.imshow("Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(20) == ord('q'):
            break
        
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
