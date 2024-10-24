import socket
import struct
import logging
import threading

logger = logging.getLogger(__name__)

class VideoSocketReceiver:
    def __init__(self, stop_event, video_frame_queue, host="0.0.0.0", port=12347):
        self.stop_event = stop_event
        self.video_frame_queue = video_frame_queue
        self.host = host
        self.port = port
        self.socket = None
        self.thread = None

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info(f"Video receiver waiting for connection on {self.host}:{self.port}")
        
        self.thread = threading.Thread(target=self.receive_video)
        self.thread.start()

    def receive_video(self):
        logger.info("Waiting for video connection...")
        conn, addr = self.socket.accept()
        logger.info(f"Video connection established with {addr}")
        
        try:
            while not self.stop_event.is_set():
                # Receba o tamanho do quadro
                size_data = conn.recv(4)
                if not size_data:
                    break
                frame_size = struct.unpack('>I', size_data)[0]

                # Receba o quadro
                frame_data = b''
                while len(frame_data) < frame_size:
                    chunk = conn.recv(frame_size - len(frame_data))
                    if not chunk:
                        break
                    frame_data += chunk

                if len(frame_data) == frame_size:
                    self.video_frame_queue.put(frame_data)
                    logger.debug(f"Received video frame of size {frame_size}")
                else:
                    logger.warning("Incomplete frame received")

        except Exception as e:
            logger.error(f"Error receiving video frame: {e}")
        finally:
            conn.close()

    def stop(self):
        self.stop_event.set()
        if self.socket:
            self.socket.close()
        if self.thread:
            self.thread.join()
        logger.info("Video receiver stopped")
