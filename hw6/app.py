"""FAU ID: alunsford2023."""

import argparse
import concurrent.futures
import http.client
import ipaddress
import os
from pathlib import Path
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import quote, urlsplit


DEFAULT_TIMEOUT = 1.5
MAX_WORKERS = 10
MAX_PORTS = 64
MAX_PAYLOADS = 64
MAX_WORDLIST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
HOST_PATTERN = re.compile(r"localhost|(?:\d{1,3}\.){3}\d{1,3}|\[?::1\]?")


class InputError(ValueError):
    """Raised when an input is outside the supported limits."""


def _validate_local_host(host):
    """Return resolved loopback addresses for a local host."""
    if not isinstance(host, str) or not HOST_PATTERN.fullmatch(host):
        raise InputError("target must be localhost or a loopback IP address")
    normalized = host.strip("[]")
    try:
        address = ipaddress.ip_address(normalized)
        if not address.is_loopback:
            raise InputError("target must resolve to a loopback address")
    except ValueError:
        if normalized != "localhost":
            raise InputError("invalid loopback IP address") from None

    try:
        results = socket.getaddrinfo(
            normalized, None, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise InputError(f"could not resolve local target: {exc}") from exc

    addresses = []
    for family, socktype, proto, _, sockaddr in results:
        address = ipaddress.ip_address(sockaddr[0])
        if not address.is_loopback:
            raise InputError("target must resolve only to loopback addresses")
        entry = (family, socktype, proto, sockaddr)
        if entry not in addresses:
            addresses.append(entry)
    if not addresses:
        raise InputError("target did not resolve to a loopback address")
    return addresses


def _parse_ports(value):
    """Parse and validate a comma-separated port list."""
    ports = []
    for item in value.split(","):
        item = item.strip()
        if not item or not item.isdecimal():
            raise InputError(f"invalid port: {item!r}")
        try:
            port = int(item)
        except ValueError:
            raise InputError("port number is too large") from None
        if not 1 <= port <= 65535:
            raise InputError(f"port out of range: {port}")
        if port not in ports:
            ports.append(port)
    if not ports:
        raise InputError("at least one port is required")
    if len(ports) > MAX_PORTS:
        raise InputError(f"no more than {MAX_PORTS} ports may be scanned")
    return ports


def _scan_port(addresses, port, timeout):
    """Check one TCP port and capture any immediately available banner."""
    last_error = None
    for family, socktype, proto, sockaddr in addresses:
        address = list(sockaddr)
        address[1] = port
        try:
            with socket.socket(family, socktype, proto) as connection:
                connection.settimeout(timeout)
                connection.connect(tuple(address))
                try:
                    banner = connection.recv(1024)
                except socket.timeout:
                    banner = b""
                text = banner.decode("utf-8", errors="replace").strip()
                return port, "open", text
        except ConnectionRefusedError:
            last_error = "connection refused"
        except socket.timeout:
            last_error = "timed out"
        except OSError as exc:
            last_error = str(exc)
    state = "closed" if last_error == "connection refused" else "unavailable"
    return port, state, last_error or ""


def _scan(args):
    addresses = _validate_local_host(args.target)
    ports = _parse_ports(args.ports)
    timeout = min(args.timeout, DEFAULT_TIMEOUT)
    workers = min(args.workers, MAX_WORKERS, len(ports))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_scan_port, addresses, port, timeout) for port in ports
        ]
        results = [future.result() for future in futures]
    for port, state, banner in sorted(results):
        line = f"{port}/tcp {state}"
        if banner:
            line += f" | {banner}"
        print(line)


def _clean_payload(payload):
    """Remove control characters from a payload."""
    return "".join(char for char in payload if char.isprintable())


def _load_payloads(args):
    if args.payloads is not None:
        source = args.payloads.split(",")
    else:
        source = _read_wordlist(args.wordlist)
    payloads = [_clean_payload(item.strip()) for item in source]
    payloads = [item for item in payloads if item]
    if not payloads:
        raise InputError("at least one non-empty payload is required")
    if len(payloads) > MAX_PAYLOADS:
        raise InputError(f"no more than {MAX_PAYLOADS} payloads are allowed")
    return payloads


def _read_wordlist(filename):
    """Read a small wordlist located beneath the current directory."""
    raw_path = os.fspath(filename)
    parts = Path(raw_path).parts
    if ".." in parts:
        raise InputError("wordlist path must not contain '..'")
    cwd = os.path.realpath(os.getcwd())
    candidate = os.path.abspath(raw_path)
    try:
        if os.path.commonpath((cwd, candidate)) != cwd:
            raise InputError("wordlist must be inside the current directory")
    except ValueError:
        raise InputError("wordlist must be inside the current directory") from None

    relative = os.path.relpath(candidate, cwd)
    current = Path(cwd)
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise InputError("wordlist paths must not contain symlinks")
    resolved = os.path.realpath(candidate)
    if os.path.commonpath((cwd, resolved)) != cwd:
        raise InputError("wordlist must be inside the current directory")
    if not os.path.isfile(resolved):
        raise InputError("wordlist must be a regular file")
    if os.path.getsize(resolved) > MAX_WORDLIST_BYTES:
        raise InputError(f"wordlist must be no larger than {MAX_WORDLIST_BYTES} bytes")
    try:
        with open(resolved, "r", encoding="utf-8") as wordlist:
            return wordlist.read().splitlines()
    except (OSError, UnicodeError) as exc:
        raise InputError(f"could not read wordlist: {exc}") from exc


def _parse_fuzz_url(value):
    """Validate a URL containing exactly one FUZZ token."""
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise InputError("URL must not contain control characters")
    if value.count("FUZZ") != 1:
        raise InputError("URL must contain exactly one FUZZ token")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise InputError(f"invalid URL: {exc}") from exc
    if parsed.scheme.lower() not in ("http", "https"):
        raise InputError("URL scheme must be http or https")
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise InputError(f"invalid URL authority: {exc}") from exc
    if not hostname or parsed.username or parsed.password:
        raise InputError("URL must contain a host and no user credentials")
    if parsed.fragment:
        raise InputError("URL fragments are not supported")
    _validate_local_host(hostname)
    if port is not None and not 1 <= port <= 65535:
        raise InputError("URL port must be between 1 and 65535")
    return parsed


def _fuzz_one(template, payload, timeout):
    encoded = quote(payload, safe="")
    url = template.replace("FUZZ", encoded)
    parsed = urlsplit(url)
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"
    port = parsed.port
    started = time.monotonic()
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme.lower() == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, port=port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read(MAX_RESPONSE_BYTES + 1)
        elapsed = (time.monotonic() - started) * 1000
        size = f">{MAX_RESPONSE_BYTES}" if len(body) > MAX_RESPONSE_BYTES else str(len(body))
        return payload, response.status, size, elapsed
    finally:
        connection.close()


def _fuzz(args):
    _parse_fuzz_url(args.url)
    payloads = _load_payloads(args)
    timeout = min(args.timeout, DEFAULT_TIMEOUT)
    for payload in payloads:
        try:
            item, status, size, elapsed = _fuzz_one(args.url, payload, timeout)
        except (OSError, http.client.HTTPException) as exc:
            print(f"{payload!r} | ERROR | {exc}")
            continue
        print(f"{item!r} | {status} | {size} bytes | {elapsed:.1f} ms")


class _DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = self.path.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        return


def _demo_security():
    """Run local checks for valid behavior and rejected inputs."""
    _validate_local_host("127.0.0.1")
    assert _parse_ports("80,443") == [80, 443]
    assert _clean_payload("safe\r\nInjected: value") == "safeInjected: value"

    rejected = (
        lambda: _validate_local_host("127.0.0.1;whoami"),
        lambda: _parse_fuzz_url("file:///etc/FUZZ"),
        lambda: _parse_fuzz_url("http://example.com/FUZZ"),
        lambda: _read_wordlist("../outside.txt"),
    )
    for check in rejected:
        try:
            check()
        except (InputError, OSError):
            continue
        raise AssertionError("an adversarial input was accepted")

    server = HTTPServer(("127.0.0.1", 0), _DemoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        address = _validate_local_host("127.0.0.1")
        port = server.server_address[1]
        result = _scan_port(address, port, DEFAULT_TIMEOUT)
        assert result[1] == "open"
        status = _fuzz_one(
            f"http://127.0.0.1:{port}/FUZZ", "demo", DEFAULT_TIMEOUT
        )
        assert status[1] == 200 and status[2] == "5"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=DEFAULT_TIMEOUT)
    print("Security demo passed.")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Loopback-only TCP scanner and HTTP path tester."
    )
    parser.add_argument(
        "--demo-security",
        action="store_true",
        help="run local functional and input-validation checks",
    )
    commands = parser.add_subparsers(dest="command")

    scan = commands.add_parser("scan", help="scan local TCP ports")
    scan.add_argument("--target", required=True, help="localhost or loopback IP")
    scan.add_argument("--ports", required=True, help="comma-separated port list")
    scan.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    scan.add_argument("--workers", type=int, default=MAX_WORKERS)

    fuzz = commands.add_parser("fuzz", help="test local HTTP paths")
    fuzz.add_argument("--url", required=True, help="URL template containing FUZZ")
    payload_source = fuzz.add_mutually_exclusive_group(required=True)
    payload_source.add_argument("--payloads", help="comma-separated payloads")
    payload_source.add_argument("--wordlist", help="wordlist file under current directory")
    fuzz.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.demo_security:
        if args.command is not None:
            parser.error("--demo-security cannot be combined with a subcommand")
        try:
            _demo_security()
        except (AssertionError, InputError, OSError, http.client.HTTPException) as exc:
            parser.exit(1, f"Security demo failed: {exc}\n")
        return 0
    if args.command is None:
        parser.error("a subcommand is required")
    if not 0 < args.timeout <= DEFAULT_TIMEOUT:
        parser.error(f"--timeout must be greater than 0 and no more than {DEFAULT_TIMEOUT}")
    if args.command == "scan" and not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    try:
        if args.command == "scan":
            _scan(args)
        else:
            _fuzz(args)
    except InputError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())