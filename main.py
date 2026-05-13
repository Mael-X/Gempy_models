import cv2
import numpy as np
import time
from kinect_capture import KinectCapture
from sandbox_mapper import SandboxMapper

def main():
    # Initialize components
    kinect = KinectCapture()
    mapper = SandboxMapper()

    # Start Kinect capture
    import threading
    capture_thread = threading.Thread(target=kinect.start)
    capture_thread.daemon = True
    capture_thread.start()

    # Wait a bit for Kinect to initialize
    time.sleep(2)

    # Create fullscreen window for projection
    cv2.namedWindow('Sandbox Projection', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Sandbox Projection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            # Get depth frame
            depth_mm = kinect.get_depth_mm()

            if depth_mm is not None:
                # Create projection image
                projection = mapper.create_projection_image(depth_mm)

                if projection is not None:
                    # Display on projector
                    cv2.imshow('Sandbox Projection', projection)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05)  # ~20 FPS

    except KeyboardInterrupt:
        pass
    finally:
        kinect.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\mael\Desktop\FabLab\main.py