"""OAuth 2.1 for MCP's HTTP transport: the handshake, by hand.

A commercial MCP server answers the very first request with

    401 + WWW-Authenticate: Bearer resource_metadata="https://…"

which is not a rejection so much as directions. Spec 2025-06-18 points
at RFC 9728 (protected-resource metadata), RFC 8414 (authorization
server metadata), RFC 7591 (dynamic client registration) and RFC 7636
(PKCE), and the walk between them is short enough to write out:

1. read ``resource_metadata`` off the challenge -> which authorization
   server guards this resource;
2. read THAT server's metadata -> where to register, authorize, and
   collect tokens;
3. register ourselves (no developer-portal visit, no client secret --
   these servers advertise ``token_endpoint_auth_methods: ["none"]``,
   i.e. a PUBLIC client, which is exactly what a local CLI is);
4. bounce the human through the browser with a PKCE challenge, catching
   the redirect on a localhost port WE are already listening on;
5. trade the code (plus the PKCE verifier) for an access token, and
   keep the refresh token for next time.

Why PKCE is not optional here: a public client has no secret, so the
authorization code is the only thing standing between an attacker who
can see the redirect and a live token. The verifier -- held only in
this process -- is what proves the code is being redeemed by whoever
asked for it.

Tokens live OUTSIDE the working directory (``~/.local/state/akshara``,
mode 0600). ``.akshara/`` is gitignored here, but a refresh token in a
project folder is one ``git add -A`` in one careless repo away from
being public, and a refresh token is a long-lived credential.

Nothing here is MCP-specific except the ``resource`` parameter; it is
plain OAuth 2.1 against httpx + stdlib, no new dependencies.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse

import httpx

#: Where tokens rest. Deliberately not the working directory.
TOKEN_DIR = Path.home() / ".local" / "state" / "akshara"
TOKEN_FILE = TOKEN_DIR / "mcp-tokens.json"

#: Refresh this long before the server would call it expired -- a token
#: that dies mid-request costs a retry and a confusing log line.
EXPIRY_SKEW_SECONDS = 60

CLIENT_NAME = "akshara"


class MCPAuthError(Exception):
    """Anything that goes wrong between the 401 and a usable token."""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def resource_metadata_url(www_authenticate: str) -> str | None:
    """Pull ``resource_metadata="…"`` out of a WWW-Authenticate header.

    Returns None when the header is missing or says nothing about where
    the metadata lives -- the caller then has a 401 it cannot act on,
    which is a better message than a guessed URL.
    """
    if not www_authenticate:
        return None
    # A regex, not a split on "," and "=": the header opens with the
    # auth SCHEME ("Bearer resource_metadata=…"), the value is a URL that
    # may itself contain "=" in a query, and other parameters may sit on
    # either side of it. Matching the parameter by name sidesteps all three.
    match = re.search(r'resource_metadata\s*=\s*(?:"([^"]*)"|([^\s,]+))',
                      www_authenticate, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1) if match.group(1) is not None else match.group(2)


def _get_json(client: httpx.Client, url: str, what: str) -> dict[str, Any]:
    try:
        response = client.get(url, timeout=15.0)
    except httpx.HTTPError as exc:
        raise MCPAuthError(f"cannot reach {what} at {url}: {exc}") from None
    if response.status_code != 200:
        raise MCPAuthError(
            f"{what} at {url} answered HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError:
        raise MCPAuthError(f"{what} at {url} is not JSON") from None


def _auth_server_metadata_urls(issuer: str) -> list[str]:
    """Candidate metadata locations for an issuer, in RFC 8414 order.

    The spec inserts ``.well-known`` BEFORE the path, which surprises
    people who expect it appended; servers in the wild answer both, so
    both get tried rather than betting on one.
    """
    parsed = urlparse(issuer)
    root = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")
    candidates = [
        f"{root}/.well-known/oauth-authorization-server{path}",
        f"{root}/.well-known/openid-configuration{path}",
        f"{root}/.well-known/oauth-authorization-server",
    ]
    if path:
        candidates.insert(2, urljoin(issuer + "/",
                                     ".well-known/oauth-authorization-server"))
    return candidates


@dataclass(slots=True)
class AuthServer:
    """The endpoints one authorization server advertises."""

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None = None
    scopes_supported: list[str] = field(default_factory=list)
    supports_s256: bool = True


def discover(resource_url: str, challenge: str, *,
             client: httpx.Client | None = None) -> tuple[AuthServer, str]:
    """From a 401 challenge to a usable AuthServer + resource id."""
    owned = client is None
    client = client or httpx.Client(follow_redirects=True)
    try:
        meta_url = resource_metadata_url(challenge)
        if meta_url is None:
            raise MCPAuthError(
                "the server asked for authentication but did not say where "
                "its metadata lives (no resource_metadata in "
                "WWW-Authenticate), so there is nothing to discover")
        resource_meta = _get_json(client, meta_url,
                                  "protected-resource metadata")
        servers = resource_meta.get("authorization_servers") or []
        if not servers:
            raise MCPAuthError(
                f"protected-resource metadata at {meta_url} names no "
                "authorization_servers")
        issuer = str(servers[0])
        resource = str(resource_meta.get("resource") or resource_url)

        last: Exception | None = None
        for candidate in _auth_server_metadata_urls(issuer):
            try:
                meta = _get_json(client, candidate,
                                 "authorization-server metadata")
                break
            except MCPAuthError as exc:
                last = exc
        else:
            raise MCPAuthError(
                f"no authorization-server metadata for {issuer}: {last}")

        for required in ("authorization_endpoint", "token_endpoint"):
            if not meta.get(required):
                raise MCPAuthError(
                    f"authorization server {issuer} advertises no "
                    f"{required}")
        methods = meta.get("code_challenge_methods_supported") or []
        return AuthServer(
            issuer=str(meta.get("issuer") or issuer),
            authorization_endpoint=str(meta["authorization_endpoint"]),
            token_endpoint=str(meta["token_endpoint"]),
            registration_endpoint=(str(meta["registration_endpoint"])
                                   if meta.get("registration_endpoint")
                                   else None),
            scopes_supported=[str(s) for s in
                              (meta.get("scopes_supported") or [])],
            supports_s256=("S256" in methods) if methods else True,
        ), resource
    finally:
        if owned:
            client.close()


# ---------------------------------------------------------------------------
# PKCE + the redirect listener
# ---------------------------------------------------------------------------


def make_pkce() -> tuple[str, str]:
    """(verifier, S256 challenge), base64url without padding per RFC 7636."""
    verifier = base64.urlsafe_b64encode(os.urandom(40)).decode().rstrip("=")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge


class _RedirectCatcher(BaseHTTPRequestHandler):
    """Record the OAUTH redirect -- and only that.

    A browser sends more than the one request you asked for: it fetches
    /favicon.ico for the page below, and may prefetch or re-request on
    its own schedule. An earlier version recorded EVERY GET, so a
    favicon (no query, therefore no ``state``) overwrote the real
    redirect and the flow died claiming a CSRF mismatch. Only a request
    actually carrying ``code`` or ``error`` counts; everything else gets
    a 404 and is forgotten.
    """

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler's name)
        from urllib.parse import parse_qs
        query = {k: v[0] for k, v in
                 parse_qs(urlparse(self.path).query).items()}
        if "code" not in query and "error" not in query:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.server.result = query  # type: ignore[attr-defined]
        # The empty data: icon is not decoration -- it stops the browser
        # requesting /favicon.ico, which is the noise the guard above
        # exists to survive. Belt and braces: either one alone suffices.
        body = (b"<html><head><link rel='icon' href='data:,'>"
                b"<title>Signed in</title></head>"
                b"<body style='font:16px system-ui;padding:3rem'>"
                b"<h2>Signed in.</h2><p>You can close this tab and go back "
                b"to the terminal.</p></body></html>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass  # the browser's chatter is not our user's business


class RedirectListener:
    """A bound localhost port waiting for the browser to come back.

    Binding happens in __init__ and the port is HELD for the whole flow.
    That ordering is not incidental: the redirect_uri has to be fixed
    before registration can name it, and a listener rebound afterwards
    could land on a different port -- registering one URI and listening
    on another, which fails at the very last step with an error about
    redirect mismatch that explains nothing.
    """

    def __init__(self, host: str = "127.0.0.1") -> None:
        self._httpd = HTTPServer((host, 0), _RedirectCatcher)
        self._httpd.result = None  # type: ignore[attr-defined]
        self.port: int = self._httpd.server_address[1]
        self.redirect_uri = f"http://{host}:{self.port}/callback"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True,
            name="mcp-oauth-redirect")
        self._thread.start()

    def wait(self, state: str, timeout: float = 300.0) -> str:
        """Block until the redirect lands; return the authorization code."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = getattr(self._httpd, "result", None)
            if result is not None:
                # The server's own complaint comes FIRST. Checking state
                # ahead of it means an error redirect that omits state --
                # plenty do -- is reported as a CSRF failure, which hides
                # the one piece of information the user needed.
                if "error" in result:
                    detail = result.get("error_description")
                    raise MCPAuthError(
                        f"authorization refused: {result['error']}"
                        + (f" ({detail})" if detail else ""))
                # Only now, with a code on the table, does state matter --
                # it guards accepting one, and there is nothing to accept
                # in the branch above.
                if result.get("state") != state:
                    arrived = ", ".join(sorted(result)) or "nothing"
                    got = result.get("state")
                    raise MCPAuthError(
                        "the redirect carried "
                        + ("no 'state' at all" if got is None
                           else "a 'state' we did not send")
                        + f" -- refusing the code. Parameters that arrived: "
                          f"{arrived}. That is the CSRF check doing its job; "
                          "start the login again")
                code = result.get("code")
                if not code:
                    raise MCPAuthError(
                        "the redirect carried neither a code nor an error; "
                        "nothing to exchange")
                return code
            time.sleep(0.1)
        raise MCPAuthError(
            "timed out waiting for the browser to come back -- the login "
            "window may have been closed before it finished")

    def close(self) -> None:
        try:
            self._httpd.shutdown()
        except Exception:
            pass
        try:
            self._httpd.server_close()
        except Exception:
            pass


def authorization_url(server: AuthServer, *, client_id: str, resource: str,
                      redirect_uri: str,
                      scope: str | None = None) -> tuple[str, str, str]:
    """Build the browser URL. Returns (url, pkce_verifier, state)."""
    verifier, challenge = make_pkce()
    state = secrets.token_urlsafe(24)
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        # RFC 8707: name WHICH resource this token is for, so a server
        # guarding several cannot hand back one that works on the others.
        "resource": resource,
    }
    if scope:
        params["scope"] = scope
    return (f"{server.authorization_endpoint}?{urlencode(params)}",
            verifier, state)


# ---------------------------------------------------------------------------
# Registration and token exchange
# ---------------------------------------------------------------------------


def register_client(server: AuthServer, redirect_uri: str, *,
                    client: httpx.Client | None = None) -> str:
    """RFC 7591 dynamic registration -> a client_id.

    This is what removes the developer-portal step entirely: the client
    introduces itself at connect time. Servers that do not offer it need
    a client_id from somewhere else, which we cannot invent.
    """
    if not server.registration_endpoint:
        raise MCPAuthError(
            f"authorization server {server.issuer} offers no dynamic "
            "registration endpoint, so a client_id has to come from its "
            "dashboard -- this client cannot make one up")
    owned = client is None
    client = client or httpx.Client(follow_redirects=True)
    try:
        payload = {
            "client_name": CLIENT_NAME,
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",  # public client
        }
        try:
            response = client.post(server.registration_endpoint,
                                   json=payload, timeout=20.0)
        except httpx.HTTPError as exc:
            raise MCPAuthError(
                f"registration at {server.registration_endpoint} "
                f"failed: {exc}") from None
        if response.status_code not in (200, 201):
            raise MCPAuthError(
                f"registration refused: HTTP {response.status_code} "
                f"{response.text[:200]!r}")
        client_id = response.json().get("client_id")
        if not client_id:
            raise MCPAuthError("registration returned no client_id")
        return str(client_id)
    finally:
        if owned:
            client.close()


def exchange_code(server: AuthServer, *, code: str, client_id: str,
                  redirect_uri: str, verifier: str, resource: str,
                  client: httpx.Client | None = None) -> dict[str, Any]:
    """Authorization code + PKCE verifier -> token response."""
    return _token_request(server, {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": verifier,
        "resource": resource,
    }, client=client)


def refresh_token(server: AuthServer, *, token: str, client_id: str,
                  resource: str,
                  client: httpx.Client | None = None) -> dict[str, Any]:
    return _token_request(server, {
        "grant_type": "refresh_token",
        "refresh_token": token,
        "client_id": client_id,
        "resource": resource,
    }, client=client)


def _token_request(server: AuthServer, form: dict[str, str], *,
                   client: httpx.Client | None) -> dict[str, Any]:
    owned = client is None
    client = client or httpx.Client(follow_redirects=True)
    try:
        try:
            response = client.post(server.token_endpoint, data=form,
                                   timeout=20.0)
        except httpx.HTTPError as exc:
            raise MCPAuthError(
                f"token endpoint {server.token_endpoint} unreachable: "
                f"{exc}") from None
        if response.status_code != 200:
            raise MCPAuthError(
                f"token endpoint refused: HTTP {response.status_code} "
                f"{response.text[:200]!r}")
        try:
            payload = response.json()
        except ValueError:
            raise MCPAuthError("token endpoint did not answer JSON") from None
        if not payload.get("access_token"):
            raise MCPAuthError("token response carried no access_token")
        return payload
    finally:
        if owned:
            client.close()


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StoredToken:
    access_token: str
    refresh_token: str | None = None
    expires_at: float | None = None
    client_id: str = ""
    issuer: str = ""
    token_endpoint: str = ""
    authorization_endpoint: str = ""
    registration_endpoint: str | None = None
    resource: str = ""

    def expired(self, *, skew: int = EXPIRY_SKEW_SECONDS) -> bool:
        """Unknown expiry counts as live: servers may omit expires_in, and
        guessing 'expired' would mean a pointless refresh on every call."""
        return self.expires_at is not None and time.time() >= (
            self.expires_at - skew)

    def as_server(self) -> AuthServer:
        return AuthServer(issuer=self.issuer,
                          authorization_endpoint=self.authorization_endpoint,
                          token_endpoint=self.token_endpoint,
                          registration_endpoint=self.registration_endpoint)


def _read_all(path: Path | None = None) -> dict[str, Any]:
    path = path or TOKEN_FILE
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}  # a corrupt store means "log in again", not a dead CLI
    return raw if isinstance(raw, dict) else {}


def load_token(server_name: str, path: Path | None = None) -> StoredToken | None:
    entry = _read_all(path).get(server_name)
    if not isinstance(entry, dict) or not entry.get("access_token"):
        return None
    known = {f for f in StoredToken.__dataclass_fields__}
    return StoredToken(**{k: v for k, v in entry.items() if k in known})


def save_token(server_name: str, token: StoredToken,
               path: Path | None = None) -> None:
    """Upsert one server's tokens, 0600, atomically.

    The mode is set on the TEMP file before the rename, so the secret is
    never briefly world-readable -- a chmod after the fact has a window.
    """
    path = path or TOKEN_FILE
    data = _read_all(path)
    data[server_name] = {k: getattr(token, k)
                         for k in StoredToken.__dataclass_fields__}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def forget_token(server_name: str, path: Path | None = None) -> bool:
    path = path or TOKEN_FILE
    data = _read_all(path)
    if server_name not in data:
        return False
    del data[server_name]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    return True


def _store_from_response(payload: dict[str, Any], *, server: AuthServer,
                         client_id: str, resource: str,
                         previous: StoredToken | None = None) -> StoredToken:
    expires_in = payload.get("expires_in")
    return StoredToken(
        access_token=str(payload["access_token"]),
        # a refresh response may omit the refresh token, meaning "keep
        # using the one you have" -- dropping it would log the user out
        # at the next expiry for no reason
        refresh_token=str(payload.get("refresh_token")
                          or (previous.refresh_token if previous else "")
                          or "") or None,
        expires_at=(time.time() + float(expires_in)) if expires_in else None,
        client_id=client_id,
        issuer=server.issuer,
        token_endpoint=server.token_endpoint,
        authorization_endpoint=server.authorization_endpoint,
        registration_endpoint=server.registration_endpoint,
        resource=resource,
    )


# ---------------------------------------------------------------------------
# The whole walk, for callers who just want a token
# ---------------------------------------------------------------------------


def login(server_name: str, resource_url: str, challenge: str, *,
          open_browser: Any = webbrowser.open,
          on_url: Any = None,
          client: httpx.Client | None = None,
          path: Path | None = None,
          timeout: float = 300.0) -> StoredToken:
    """Discovery -> registration -> browser -> token, saved and returned.

    ``on_url`` is handed the authorization URL before the browser opens,
    so a caller can print it: headless boxes and stubborn default-browser
    settings both end with the human pasting it somewhere themselves.
    """
    owned = client is None
    client = client or httpx.Client(follow_redirects=True)
    listener = RedirectListener()
    try:
        server, resource = discover(resource_url, challenge, client=client)
        # Registration names the redirect_uri, so the port must already be
        # ours -- the listener bound above and stays bound until we are done.
        client_id = register_client(server, listener.redirect_uri,
                                    client=client)
        url, verifier, state = authorization_url(
            server, client_id=client_id, resource=resource,
            redirect_uri=listener.redirect_uri,
            scope=" ".join(server.scopes_supported) or None)
        if on_url is not None:
            on_url(url)
        if open_browser is not None:
            open_browser(url)
        code = listener.wait(state, timeout=timeout)
        payload = exchange_code(server, code=code, client_id=client_id,
                                redirect_uri=listener.redirect_uri,
                                verifier=verifier, resource=resource,
                                client=client)
        token = _store_from_response(payload, server=server,
                                     client_id=client_id, resource=resource)
        save_token(server_name, token, path)
        return token
    finally:
        listener.close()
        if owned:
            client.close()


def access_token_for(server_name: str, *, path: Path | None = None,
                     client: httpx.Client | None = None,
                     force: bool = False) -> str | None:
    """A live access token for this server, refreshing if it has expired.

    Returns None when there is nothing stored -- the caller then has a
    plain unauthenticated attempt to make, which is the right thing for
    every server that needs no login at all. ``force`` refreshes even
    when the token still looks live, which is what a 401 means: the
    server disagrees with our arithmetic, and the server is right.
    """
    stored = load_token(server_name, path)
    if stored is None:
        return None
    if not force and not stored.expired():
        return stored.access_token
    if not stored.refresh_token:
        return None  # expired with no way back: caller must log in again
    payload = refresh_token(stored.as_server(), token=stored.refresh_token,
                            client_id=stored.client_id,
                            resource=stored.resource, client=client)
    refreshed = _store_from_response(payload, server=stored.as_server(),
                                     client_id=stored.client_id,
                                     resource=stored.resource,
                                     previous=stored)
    save_token(server_name, refreshed, path)
    return refreshed.access_token
