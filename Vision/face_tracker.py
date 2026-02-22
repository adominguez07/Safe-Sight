import cv2
import mediapipe as mp


class FaceTracker:
    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 0) -> None:
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def detect_and_draw(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb_frame)

        if not results.detections:
            return frame

        height, width, _ = frame.shape
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x1 = max(int(bbox.xmin * width), 0)
            y1 = max(int(bbox.ymin * height), 0)
            x2 = min(int((bbox.xmin + bbox.width) * width), width)
            y2 = min(int((bbox.ymin + bbox.height) * height), height)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame

    def close(self) -> None:
        self._detector.close()


def main(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    tracker = FaceTracker()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = tracker.detect_and_draw(frame)
            cv2.imshow("Safe-Sight", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
