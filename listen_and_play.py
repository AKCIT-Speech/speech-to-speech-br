import socket
import threading
from queue import Queue
from dataclasses import dataclass, field
import sounddevice as sd
from transformers import HfArgumentParser
import cv2
import struct
import numpy as np


@dataclass
class ListenAndPlayArguments:
    send_rate: int = field(default=16000, metadata={"help": "In Hz. Default is 16000."})
    recv_rate: int = field(default=16000, metadata={"help": "In Hz. Default is 16000."})
    list_play_chunk_size: int = field(
        default=1024,
        metadata={"help": "The size of data chunks (in bytes). Default is 1024."},
    )
    host: str = field(
        default="localhost",
        metadata={
            "help": "The hostname or IP address for listening and playing. Default is 'localhost'."
        },
    )
    send_port: int = field(
        default=12345,
        metadata={"help": "The network port for sending data. Default is 12345."},
    )
    recv_port: int = field(
        default=12346,
        metadata={"help": "The network port for receiving data. Default is 12346."},
    )
    video_port: int = field(
        default=12347,
        metadata={"help": "The network port for sending video data. Default is 5001."},
    )


def listen_and_play(
    send_rate=16000,
    recv_rate=44100,
    list_play_chunk_size=1024,
    host="localhost",
    send_port=12345,
    recv_port=12346,
    video_port=12347,
):
    video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    video_socket.connect((host, video_port))
    
    send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    send_socket.connect((host, send_port))

    recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    recv_socket.connect((host, recv_port))

    print("Recording, streaming audio and video...")

    stop_event = threading.Event()
    recv_queue = Queue()
    send_queue = Queue()

    def callback_recv(outdata, frames, time, status):
        if not recv_queue.empty():
            data = recv_queue.get()
            outdata[: len(data)] = data
            outdata[len(data) :] = b"\x00" * (len(outdata) - len(data))
        else:
            outdata[:] = b"\x00" * len(outdata)

    def callback_send(indata, frames, time, status):
        if recv_queue.empty():
            data = bytes(indata)
            send_queue.put(data)

    def send(stop_event, send_queue):
        while not stop_event.is_set():
            data = send_queue.get()
            send_socket.sendall(data)

    def recv(stop_event, recv_queue):
        def receive_full_chunk(conn, chunk_size):
            data = b""
            while len(data) < chunk_size:
                packet = conn.recv(chunk_size - len(data))
                if not packet:
                    return None  # Connection has been closed
                data += packet
            return data

        while not stop_event.is_set():
            data = receive_full_chunk(recv_socket, list_play_chunk_size * 2)
            if data:
                recv_queue.put(data)

    def send_video(stop_event, video_socket):
        cap = cv2.VideoCapture(0)  # Abre a webcam
        while not stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                # Redimensiona o frame para 224x224 (ou outro tamanho desejado)
                frame = cv2.resize(frame, (224, 224))
                # Converte para escala de cinza
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Serializa o frame
                frame_bytes = frame.tobytes()
                # Envia o tamanho do frame seguido pelo frame
                video_socket.sendall(struct.pack('>I', len(frame_bytes)))
                video_socket.sendall(frame_bytes)
        cap.release()

    try:
        send_stream = sd.RawInputStream(
            samplerate=send_rate,
            channels=1,
            dtype="int16",
            blocksize=list_play_chunk_size,
            callback=callback_send,
        )
        recv_stream = sd.RawOutputStream(
            samplerate=recv_rate,
            channels=1,
            dtype="int16",
            blocksize=list_play_chunk_size,
            callback=callback_recv,
        )
        threading.Thread(target=send_stream.start).start()
        threading.Thread(target=recv_stream.start).start()

        send_thread = threading.Thread(target=send, args=(stop_event, send_queue))
        send_thread.start()
        recv_thread = threading.Thread(target=recv, args=(stop_event, recv_queue))
        recv_thread.start()
        video_thread = threading.Thread(target=send_video, args=(stop_event, video_socket))
        video_thread.start()

        input("Pressione Enter para parar...")

    except KeyboardInterrupt:
        print("Transmissão finalizada.")

    finally:
        stop_event.set()
        recv_thread.join()
        send_thread.join()
        video_thread.join()
        send_socket.close()
        recv_socket.close()
        video_socket.close()
        print("Conexão fechada.")


if __name__ == "__main__":
    parser = HfArgumentParser((ListenAndPlayArguments,))
    (listen_and_play_kwargs,) = parser.parse_args_into_dataclasses()
    listen_and_play(**vars(listen_and_play_kwargs))
