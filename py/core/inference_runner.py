def run_inference(frame_source, frame_sink=None, backend=None, stop_fn=lambda: False, log_fn=print):
    log_fn("inference runner started")
    if frame_sink is None:
        frame_sink = lambda f: None
    
    try:
        for frame in frame_source():
            if stop_fn and stop_fn():
                log_fn("stop requested")
                break
            out = backend.step(frame)
            frame_sink(out)
    except StopIteration:
        # frame_sink or frame_source can raise StopIteration to stop gracefully
        log_fn("inference stopped by sink")
    except Exception as e:
        log_fn("inference runner error:", e)
    finally:
        log_fn("inference runner exiting")
