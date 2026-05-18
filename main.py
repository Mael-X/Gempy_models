import cv2
import numpy as np
import time
from kinect_capture import KinectCapture
from sandbox_mapper import SandboxMapper
import threading

def main():
    # Initialize components
    kinect = KinectCapture()
    mapper = SandboxMapper()

    # Restrict processing to a smaller Kinect ROI for better responsiveness.
    # ROI is (x0, y0, x1, y1) in Kinect pixel coordinates.
    mapper.set_roi((160, 120, 480, 360))

    # Start Kinect capture
    capture_thread = threading.Thread(target=kinect.start)
    capture_thread.daemon = True
    capture_thread.start()

    # Wait a bit for Kinect to initialize
    time.sleep(2)

    # Create fullscreen window for projection
    cv2.namedWindow('Sandbox Projection', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Sandbox Projection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def clamp_roi(roi, width, height):
        x0, y0, x1, y1 = roi
        x0 = max(0, min(x0, width - 1))
        y0 = max(0, min(y0, height - 1))
        x1 = max(x0 + 1, min(x1, width))
        y1 = max(y0 + 1, min(y1, height))
        return x0, y0, x1, y1

    def adjust_roi(roi, dx0=0, dy0=0, dx1=0, dy1=0):
        if roi is None:
            roi = (0, 0, 640, 480)
        x0, y0, x1, y1 = roi
        x0 += dx0
        y0 += dy0
        x1 += dx1
        y1 += dy1
        return clamp_roi((x0, y0, x1, y1), 640, 480)

    try:
        while True:
            # Get depth frame
            depth_mm = kinect.get_depth_mm()

            if depth_mm is not None:
                # Create projection image
                projection = mapper.create_projection_image(depth_mm)

                # Draw the ROI for live tuning
                roi = mapper.get_roi()
                if roi is not None and projection is not None:
                    x0, y0, x1, y1 = roi
                    cv2.rectangle(projection, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)

                if projection is not None:
                    # Display on projector
                    cv2.imshow('Sandbox Projection', projection)

            # Check for quit and ROI adjustment keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                mapper.clear_roi()
                print('ROI cleared: full frame')
            elif key == ord('t'):
                default_roi = (160, 120, 480, 360)
                mapper.set_roi(default_roi)
                print(f'ROI reset to {default_roi}')
            elif key == ord('q'):
                mapper.set_roi(adjust_roi(mapper.get_roi(), dx0=-16, dx1=-16))
            elif key == ord('d'):
                mapper.set_roi(adjust_roi(mapper.get_roi(), dx0=16, dx1=16))
            elif key == ord('z'):
                mapper.set_roi(adjust_roi(mapper.get_roi(), dy0=-16, dy1=-16))
            elif key == ord('s'):
                mapper.set_roi(adjust_roi(mapper.get_roi(), dy0=16, dy1=16))
            elif key == ord('z'):
                mapper.set_roi(adjust_roi(mapper.get_roi(), dx0=16, dy0=16, dx1=-16, dy1=-16))
            elif key == ord('x'):
                mapper.set_roi(adjust_roi(mapper.get_roi(), dx0=-16, dy0=-16, dx1=16, dy1=16))

            time.sleep(0.0016)  # ~60 FPS

    except KeyboardInterrupt:
        pass
    finally:
        kinect.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()