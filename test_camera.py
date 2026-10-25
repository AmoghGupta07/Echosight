import cv2

for backend_name, backend in [
    ("DirectShow", cv2.CAP_DSHOW),
    ("Media Foundation", cv2.CAP_MSMF),
]:
    print(f"\nTrying {backend_name}...")

    cap = cv2.VideoCapture(0, backend)

    if not cap.isOpened():
        print("Could not open camera")
        cap.release()
        continue

    # Start with a common webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    for attempt in range(10):
        ok, frame = cap.read()

        if ok and frame is not None:
            print(f"SUCCESS: {frame.shape}")
            cv2.imwrite("test_frame.jpg", frame)
            print("Saved test_frame.jpg")
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit(0)

        print(f"Read failed: {attempt + 1}")

    cap.release()

cv2.destroyAllWindows()
print("Both backends failed. Check camera access and supported formats.")
