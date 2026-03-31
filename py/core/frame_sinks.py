import cv2

# cli sink for opencv display, with quit on 'q' and resizing for better performance
def opencv_sink(window_name='susieML CLI'):
    def sink(frame):
        width = int(frame.shape[1] * 0.75)
        length = int(frame.shape[0] * 0.75)
        frame = cv2.resize(frame, (width, length))
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise StopIteration
    return sink

