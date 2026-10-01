import dataclasses
import http.server
import json
import queue
import threading


@dataclasses.dataclass
class TraceStep:
    stage: str
    step_index: int
    forced: bool
    chosen_id: int
    chosen_text: str
    prompt: str
    _time: str
    _id: str
    answer: str | None = None


class VisQueue:
    """Builds a queue for visualizing generation steps.
    steps: list[TraceStep] is the full history of steps, and each new step is
    appended to it.
    _subscribers: list[queue.Queue[TraceStep | None]] is a list of queues for
    each connected browser. Each queue receives new TraceStep objects as they
    are appended to steps, and receives None when generation is finished.
    _has_subscriber: threading.Event is set when at least one browser has
    connected, so the generator can wait for a viewer to be ready before
    starting.
    _lock: threading.Lock protects symultanic access to _subscribers.
    """

    def __init__(self) -> None:
        self.steps: list[TraceStep] = []
        self._subscribers: list[queue.Queue[TraceStep | None]] = []
        self._has_subscriber = threading.Event()
        self._lock = threading.Lock()

    def append(self, step: TraceStep) -> None:
        self.steps.append(step)
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            q.put(step)

    def subscribe(self) -> queue.Queue[TraceStep | None]:
        q: queue.Queue[TraceStep | None] = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        self._has_subscriber.set()
        return q

    def unsubscribe(self, q: queue.Queue[TraceStep | None]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def wait_for_viewer(self, timeout: float | None = 30.0) -> bool:
        return self._has_subscriber.wait(timeout)

    def finish(self) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            q.put(None)


with open("src/frontend/viewer.html", 'r') as f:
    _VIEWER_HTML = f.read()


def _make_handler(live_trace: VisQueue) -> type:
    """Creates a custom HTTP request handler class with the VisQueue instance.
    Parameters
    ----------
    live_trace: VisQueue
        The VisQueue instance to use for sending TraceStep objects to the
        connected browsers.
    Returns
    -------
    type
        a subclass of http.server.BaseHTTPRequestHandler that serves
        the viewer with custom GET and stream endpoint.
    """
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            pass  # silence default per-request console logging

        def do_GET(self) -> None:
            if self.path == "/":
                body = _VIEWER_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = live_trace.subscribe()
                try:
                    while True:
                        step = q.get()
                        if step is None:
                            self.wfile.write(b"event: done\ndata: {}\n\n")
                            self.wfile.flush()
                            break
                        payload = json.dumps(dataclasses.asdict(step))
                        self.wfile.write(
                            f"data: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    live_trace.unsubscribe(q)
                return

            self.send_response(404)
            self.end_headers()

    return Handler


def start_trace_server(
        live_trace: VisQueue,
        port: int) -> http.server.ThreadingHTTPServer:
    """Starts the viewer server on a background thread and returns it --
    daemon for safety.
    Parameters
    ----------
    live_trace: VisQueue
    port: int, port to listen on
    Returns
    -------
    http.server.ThreadingHTTPServer
    the server object to call .shutdown() for double safety.
    """
    handler_cls = _make_handler(live_trace)
    server = http.server.ThreadingHTTPServer(("localhost", port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
