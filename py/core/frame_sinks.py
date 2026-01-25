import cv2

def opencv_sink(window_name='susieML v1.0.0'):
    def sink(frame):
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise StopIteration
    return sink

