#!/usr/bin/env python3
"""gpu-gateway: VRAM time-sharing front proxy for ftserve + ComfyUI on one GPU.

Owns the two well-known ports and moves the real backends one port up:

    LLM clients (:8010)  -> gateway -> ftserve (:8011)
    ComfyUI   (:8188)    -> gateway -> ComfyUI (:8191)

Rules (all timings via env, seconds):
  * An inference request wakes ftserve (systemctl start + readiness poll) and is
    held until ready; /v1/stats & /health are answered locally and never wake.
  * ftserve idles (no in-flight requests, no activity) for FT_IDLE -> stopped.
  * A ComfyUI /prompt submission waits for ftserve to drain (COMFY_WAIT_CAP),
    then stops it before the job is let through.
  * ComfyUI queue empty for COMFY_FREE_IDLE -> POST /free {"unload_all": true}.

Proxying is raw-TCP: only the first request head is parsed for routing, then
bytes are pumped in both directions, so SSE and WebSockets pass through as-is.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request

FT_PORT = int(os.environ.get("FT_PORT", 8010))
FT_UP = ("127.0.0.1", int(os.environ.get("FT_UP_PORT", 8011)))
COMFY_PORT = int(os.environ.get("COMFY_PORT", 8188))
COMFY_UP = ("127.0.0.1", int(os.environ.get("COMFY_UP_PORT", 8191)))
FT_IDLE = float(os.environ.get("FT_IDLE", 600))
COMFY_FREE_IDLE = float(os.environ.get("COMFY_FREE_IDLE", 300))
COMFY_WAIT_CAP = float(os.environ.get("COMFY_WAIT_CAP", 600))
FT_WAKE_CAP = float(os.environ.get("FT_WAKE_CAP", 300))
GOV_TICK = float(os.environ.get("GOV_TICK", 30))
QUEUE_POLL = float(os.environ.get("QUEUE_POLL", 5))

# The paths the gateway answers itself so monitoring never wakes the model.
_LOCAL_PATHS = ("/v1/stats", "/health", "/healthz", "/gpu-gateway/status")

_state: dict = {
    "ft": "unknown",  # up | down | waking
    "ft_last_active": time.time(),
    "ft_inflight": 0,
    "wake_task": None,
    "last_stats": None,
    "comfy_last_active": time.time(),
    "comfy_freed": False,
    "started": time.time(),
}


def log(msg: str) -> None:
    print(f"[gpu-gateway {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ upstream helpers
def _http_json(url: str, method: str = "GET", body: bytes | None = None, timeout: float = 3.0):
    """Blocking JSON call for use via asyncio.to_thread; None on any failure."""
    try:
        req = urllib.request.Request(url, data=body, method=method)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return None


def _ft_ready(health: dict | None) -> bool:
    """/health is the authoritative lifecycle signal: loading -> ok -> error."""
    return bool(health) and health.get("status") == "ok"


async def _ft_probe(ready_only: bool = False) -> dict | None:
    """Poll /health (lifecycle). Also refresh the cached /v1/stats when reachable."""
    health = await asyncio.to_thread(
        _http_json, f"http://127.0.0.1:{FT_UP[1]}/health"
    )
    stats = await asyncio.to_thread(
        _http_json, f"http://127.0.0.1:{FT_UP[1]}/v1/stats"
    )
    if stats is not None:
        _state["last_stats"] = stats
    if health is None:
        return None
    if ready_only and not _ft_ready(health):
        prog = health.get("progress") or {}
        total = prog.get("total_bytes") or 0
        done = prog.get("done_bytes") or 0
        pct = f" {done * 100 // total}%" if total else ""
        _state["load_note"] = f"{health.get('status')}/{health.get('phase')}{pct}"
        return None
    return health


async def _systemctl(*args: str) -> bool:
    proc = await asyncio.create_subprocess_exec(
        "systemctl", *args,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    rc = await proc.wait()
    return rc == 0


async def _wake_ft() -> bool:
    """single-flight: start ftserve and wait until /v1/stats answers."""
    _state["ft"] = "waking"
    log("waking ftserve (systemctl start)")
    if not await _systemctl("start", "ftserve"):
        _state["ft"] = "down"
        log("systemctl start ftserve failed")
        return False
    deadline = time.monotonic() + FT_WAKE_CAP
    last_note = 0.0
    while time.monotonic() < deadline:
        if await _ft_probe(ready_only=True) is not None:
            _state["ft"] = "up"
            _state["ft_last_active"] = time.time()
            log("ftserve is up")
            return True
        if time.monotonic() - last_note > 15:
            log(f"waking... {_state.get('load_note', 'connecting')}")
            last_note = time.monotonic()
        await asyncio.sleep(2)
    _state["ft"] = "down"
    log(f"ftserve did not become ready within {FT_WAKE_CAP}s")
    return False


async def ensure_ft() -> bool:
    """Make sure ftserve is serving; concurrent callers share one wake."""
    if _state["ft"] == "up" and await _ft_probe(ready_only=True) is not None:
        return True
    if _state["wake_task"] is None:
        _state["wake_task"] = asyncio.create_task(_wake_ft())
        _state["wake_task"].add_done_callback(lambda t: _state.__setitem__("wake_task", None))
    try:
        return await asyncio.wait_for(asyncio.shield(_state["wake_task"]), FT_WAKE_CAP + 5)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        return False


async def ensure_gpu_for_comfy() -> bool:
    """Stop ftserve (after draining) so ComfyUI can take the GPU."""
    deadline = time.monotonic() + COMFY_WAIT_CAP
    while time.monotonic() < deadline:
        if _state["ft"] == "down":
            return True
        reachable = await _ft_probe() is not None
        inflight = _state["ft_inflight"]
        stats = _state["last_stats"] or {}
        active = int((stats.get("requests") or {}).get("active") or 0)
        if _state["ft"] != "waking" and reachable and inflight == 0 and active == 0:
            log("parking ftserve for comfy job")
            if await _systemctl("stop", "ftserve"):
                _state["ft"] = "down"
                return True
            log("systemctl stop ftserve failed; retrying")
        await asyncio.sleep(2)
    log(f"comfy job timed out waiting for ftserve drain ({COMFY_WAIT_CAP}s)")
    return False


# ------------------------------------------------------------------ tunneling
async def _pump(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await src.read(65536)
            if not data:
                break
            dst.write(data)
            await dst.drain()
    except Exception:
        pass
    finally:
        try:
            dst.close()
        except Exception:
            pass


async def _tunnel(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                  head: bytes, rest: bytes, upstream: tuple[str, int]) -> None:
    try:
        up_r, up_w = await asyncio.wait_for(
            asyncio.open_connection(*upstream), timeout=10
        )
    except Exception:
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    up_w.write(head)
    if rest:
        up_w.write(rest)
    await asyncio.gather(_pump(reader, up_w), _pump(up_r, writer))


async def _read_head(reader: asyncio.StreamReader) -> tuple[bytes, bytes, str, str]:
    """Read until end of headers; return (head, leftover, method, path)."""
    buf = b""
    while b"\r\n\r\n" not in buf and len(buf) < 65536:
        chunk = await reader.read(4096)
        if not chunk:
            break
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    try:
        line = head.split(b"\r\n", 1)[0].decode("latin-1")
        method, path, _ = line.split(" ", 2)
    except Exception:
        return head, rest, "", ""
    return head + b"\r\n\r\n", rest, method.upper(), path.split("?", 1)[0]


def _reply(writer: asyncio.StreamWriter, status: str, payload: dict | str) -> None:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    data = body.encode()
    writer.write(
        f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data
    )


# ------------------------------------------------------------------ handlers
async def _handle_llm(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        head, rest, method, path = await asyncio.wait_for(_read_head(reader), timeout=30)
        if not method:
            writer.close()
            return
        if path in _LOCAL_PATHS:
            _reply(writer, "200 OK", _gateway_status())
            await writer.drain()
            writer.close()
            return
        _state["ft_inflight"] += 1
        try:
            if not await ensure_ft():
                _reply(writer, "503 Service Unavailable",
                       {"error": f"ftserve wake failed or timed out (> {FT_WAKE_CAP}s); retry soon"})
                await writer.drain()
                writer.close()
                return
            await _tunnel(reader, writer, head, rest, FT_UP)
        finally:
            _state["ft_inflight"] -= 1
            _state["ft_last_active"] = time.time()
    except Exception as exc:  # noqa: BLE001
        log(f"llm handler error: {exc!r}")
        try:
            writer.close()
        except Exception:
            pass


async def _handle_comfy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        head, rest, method, path = await asyncio.wait_for(_read_head(reader), timeout=30)
        if not method:
            writer.close()
            return
        _state["comfy_last_active"] = time.time()
        _state["comfy_freed"] = False
        if method == "POST" and path in ("/prompt", "/api/prompt"):
            if not await ensure_gpu_for_comfy():
                _reply(writer, "503 Service Unavailable",
                       {"error": f"ftserve did not free the GPU within {COMFY_WAIT_CAP}s; retry"})
                await writer.drain()
                writer.close()
                return
        await _tunnel(reader, writer, head, rest, COMFY_UP)
    except Exception as exc:  # noqa: BLE001
        log(f"comfy handler error: {exc!r}")
        try:
            writer.close()
        except Exception:
            pass


def _gateway_status() -> dict:
    return {
        "gpu_gateway": {
            "ft_state": _state["ft"],
            "ft_inflight": _state["ft_inflight"],
            "ft_idle_s": round(time.time() - _state["ft_last_active"], 1),
            "ft_idle_target_s": FT_IDLE,
            "comfy_freed": _state["comfy_freed"],
            "uptime_s": round(time.time() - _state["started"], 1),
        },
        "upstream_stats": _state["last_stats"],
    }


# ------------------------------------------------------------------ governor
async def _governor() -> None:
    while True:
        await asyncio.sleep(GOV_TICK)
        try:
            if (
                _state["ft"] == "up"
                and _state["ft_inflight"] == 0
                and _state["wake_task"] is None
                and time.time() - _state["ft_last_active"] > FT_IDLE
            ):
                await _ft_probe()
                stats = _state["last_stats"] or {}
                active = int((stats.get("requests") or {}).get("active") or 0)
                if active == 0:
                    log(f"ftserve idle > {FT_IDLE}s; parking to free VRAM")
                    if await _systemctl("stop", "ftserve"):
                        _state["ft"] = "down"
            if (
                not _state["comfy_freed"]
                and time.time() - _state["comfy_last_active"] > COMFY_FREE_IDLE
            ):
                log("comfy idle; sending /free unload_all")
                await asyncio.to_thread(
                    _http_json, f"http://127.0.0.1:{COMFY_UP[1]}/free", "POST",
                    b'{"unload_all": true}', 10.0,
                )
                _state["comfy_freed"] = True
        except Exception as exc:  # noqa: BLE001
            log(f"governor error: {exc!r}")


async def _comfy_queue_watch() -> None:
    """Refresh comfy activity while jobs run or wait, so /free stays away."""
    while True:
        await asyncio.sleep(QUEUE_POLL)
        try:
            q = await asyncio.to_thread(
                _http_json, f"http://127.0.0.1:{COMFY_UP[1]}/queue"
            )
            if q is None:
                continue
            busy = len(q.get("QueueRunning") or []) + len(q.get("QueuePending") or [])
            if busy:
                _state["comfy_last_active"] = time.time()
                _state["comfy_freed"] = False
        except Exception:
            pass


async def main() -> None:
    # initial state probe (no wake): "up" only when /health already says ok
    if await _ft_probe(ready_only=True) is not None:
        _state["ft"] = "up"
    else:
        _state["ft"] = "down"
    log(f"start: ft={_state['ft']} llm :{FT_PORT}->{FT_UP[1]} comfy :{COMFY_PORT}->{COMFY_UP[1]} "
        f"idle={FT_IDLE}s comfy_free={COMFY_FREE_IDLE}s wait_cap={COMFY_WAIT_CAP}s")
    asyncio.create_task(_governor())
    asyncio.create_task(_comfy_queue_watch())
    llm_srv = await asyncio.start_server(_handle_llm, "0.0.0.0", FT_PORT)
    comfy_srv = await asyncio.start_server(_handle_comfy, "0.0.0.0", COMFY_PORT)
    async with llm_srv, comfy_srv:
        await asyncio.gather(llm_srv.serve_forever(), comfy_srv.serve_forever())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
