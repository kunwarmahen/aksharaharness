"""OAuth 2.1 for MCP: the whole walk, offline.

Every network hop goes through httpx.MockTransport against a fake
authorization server, but the REDIRECT half is real -- a genuine
localhost socket, a genuine browser-shaped GET -- because that is the
part where a bound port, a state parameter and a one-shot handler have
to agree, and a mock would just report that they did.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import threading
import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from akshara.mcp_oauth import (
    AuthServer,
    MCPAuthError,
    RedirectListener,
    StoredToken,
    access_token_for,
    authorization_url,
    discover,
    exchange_code,
    forget_token,
    load_token,
    login,
    make_pkce,
    register_client,
    resource_metadata_url,
    save_token,
)

RESOURCE = "https://srv.example/mcp"
META_URL = "https://srv.example/.well-known/oauth-protected-resource/mcp"
CHALLENGE = f'Bearer resource_metadata="{META_URL}"'

AUTH_META = {
    "issuer": RESOURCE,
    "authorization_endpoint": "https://idp.example/oauth",
    "token_endpoint": "https://idp.example/token",
    "registration_endpoint": "https://srv.example/register",
    "code_challenge_methods_supported": ["S256"],
    "scopes_supported": ["internal"],
    "grant_types_supported": ["authorization_code", "refresh_token"],
    "token_endpoint_auth_methods_supported": ["none"],
}


def fake_idp(*, token_payload=None, registration=None, seen=None):
    """A MockTransport handler standing in for the whole vendor side."""
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        path = request.url.path
        if path.endswith("oauth-protected-resource/mcp"):
            return httpx.Response(200, json={
                "resource": RESOURCE,
                "authorization_servers": [RESOURCE],
                "bearer_methods_supported": ["header"],
            })
        if ".well-known/oauth-authorization-server" in path:
            return httpx.Response(200, json=AUTH_META)
        if path.endswith("/register"):
            return httpx.Response(
                201, json=registration or {"client_id": "cid-123"})
        if path.endswith("/token"):
            return httpx.Response(200, json=token_payload or {
                "access_token": "at-1", "refresh_token": "rt-1",
                "expires_in": 3600, "token_type": "Bearer"})
        return httpx.Response(404, text="nope")
    return handler


def idp_client(**kwargs) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(fake_idp(**kwargs)))


class TestChallengeParsing:
    """The 401 header is the only map to everything else."""

    @pytest.mark.parametrize("header,expected", [
        (CHALLENGE, META_URL),
        (f'Bearer realm="x", resource_metadata="{META_URL}"', META_URL),
        (f"Bearer resource_metadata={META_URL}", META_URL),      # unquoted
        ('Bearer error="invalid_token"', None),
        ("", None),
    ])
    def test_finds_the_metadata_url(self, header, expected):
        assert resource_metadata_url(header) == expected

    def test_scheme_prefix_does_not_shadow_the_parameter(self):
        """A naive split on '=' reads the key as 'Bearer resource_metadata'
        and finds nothing -- which is how this got written wrong once."""
        assert resource_metadata_url(CHALLENGE) == META_URL


class TestPKCE:
    def test_challenge_is_the_sha256_of_the_verifier(self):
        verifier, challenge = make_pkce()
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("ascii")).digest()
        ).decode().rstrip("=")
        assert challenge == expected

    def test_unpadded_and_long_enough_for_rfc7636(self):
        verifier, challenge = make_pkce()
        assert "=" not in verifier and "=" not in challenge
        assert 43 <= len(verifier) <= 128

    def test_two_calls_never_agree(self):
        assert make_pkce()[0] != make_pkce()[0]


class TestDiscovery:
    def test_walks_challenge_to_endpoints(self):
        with idp_client() as client:
            server, resource = discover(RESOURCE, CHALLENGE, client=client)
        assert server.authorization_endpoint == "https://idp.example/oauth"
        assert server.token_endpoint == "https://idp.example/token"
        assert server.registration_endpoint == "https://srv.example/register"
        assert server.scopes_supported == ["internal"]
        assert resource == RESOURCE

    def test_challenge_without_metadata_is_a_dead_end_we_name(self):
        with idp_client() as client:
            with pytest.raises(MCPAuthError, match="did not say where"):
                discover(RESOURCE, 'Bearer realm="x"', client=client)

    def test_metadata_naming_no_authorization_server_is_refused(self):
        def handler(request):
            return httpx.Response(200, json={"resource": RESOURCE})
        client = httpx.Client(transport=httpx.MockTransport(handler))
        with client:
            with pytest.raises(MCPAuthError, match="no authorization_servers"):
                discover(RESOURCE, CHALLENGE, client=client)


class TestRegistration:
    def test_registers_as_a_public_client(self):
        seen: list[httpx.Request] = []
        with idp_client(seen=seen) as client:
            server, _ = discover(RESOURCE, CHALLENGE, client=client)
            cid = register_client(server, "http://127.0.0.1:9/callback",
                                  client=client)
        assert cid == "cid-123"
        body = json.loads(seen[-1].content)
        # no secret to keep, which is the only honest posture for a
        # local CLI that ships its source
        assert body["token_endpoint_auth_method"] == "none"
        assert body["redirect_uris"] == ["http://127.0.0.1:9/callback"]
        assert "refresh_token" in body["grant_types"]

    def test_without_an_endpoint_we_say_so_rather_than_invent_an_id(self):
        server = AuthServer(issuer="i", authorization_endpoint="a",
                            token_endpoint="t", registration_endpoint=None)
        with pytest.raises(MCPAuthError, match="no dynamic registration"):
            register_client(server, "http://127.0.0.1:9/callback")


class TestAuthorizationURL:
    def test_carries_pkce_state_and_the_resource(self):
        server = AuthServer(issuer="i",
                            authorization_endpoint="https://idp.example/oauth",
                            token_endpoint="t")
        url, verifier, state = authorization_url(
            server, client_id="cid", resource=RESOURCE,
            redirect_uri="http://127.0.0.1:9/callback", scope="internal")
        q = parse_qs(urlparse(url).query)
        assert q["code_challenge_method"] == ["S256"]
        assert q["client_id"] == ["cid"]
        assert q["state"] == [state]
        assert q["resource"] == [RESOURCE]       # RFC 8707
        assert q["scope"] == ["internal"]
        assert q["code_challenge"] != [verifier]  # the challenge, not the key


class TestRedirectListener:
    """A real socket: the port has to be held, and the state checked."""

    def _hit(self, listener: RedirectListener, query: str) -> None:
        httpx.get(f"http://127.0.0.1:{listener.port}/callback?{query}",
                  timeout=5.0)

    def test_code_comes_back(self):
        listener = RedirectListener()
        try:
            threading.Timer(0.1, self._hit,
                            (listener, "code=abc&state=st")).start()
            assert listener.wait("st", timeout=5.0) == "abc"
        finally:
            listener.close()

    def test_wrong_state_is_refused(self):
        listener = RedirectListener()
        try:
            threading.Timer(0.1, self._hit,
                            (listener, "code=abc&state=ATTACKER")).start()
            with pytest.raises(MCPAuthError,
                               match="'state' we did not send"):
                listener.wait("st", timeout=5.0)
        finally:
            listener.close()

    def test_an_error_redirect_reports_the_reason(self):
        listener = RedirectListener()
        try:
            threading.Timer(0.1, self._hit, (
                listener,
                "error=access_denied&error_description=user+said+no"
                "&state=st")).start()
            with pytest.raises(MCPAuthError, match="access_denied"):
                listener.wait("st", timeout=5.0)
        finally:
            listener.close()

    def test_a_favicon_request_does_not_clobber_the_redirect(self):
        """What actually broke in the field: the browser fetches
        /favicon.ico for our 'Signed in.' page, and an earlier handler
        recorded every GET -- so a query-less request overwrote the real
        redirect and the flow died claiming CSRF."""
        listener = RedirectListener()
        try:
            def noise_then_code():
                httpx.get(f"http://127.0.0.1:{listener.port}/favicon.ico",
                          timeout=5.0)
                httpx.get(f"http://127.0.0.1:{listener.port}/callback"
                          "?code=abc&state=st", timeout=5.0)
            threading.Timer(0.05, noise_then_code).start()
            assert listener.wait("st", timeout=5.0) == "abc"
        finally:
            listener.close()

    def test_noise_after_the_code_is_ignored_too(self):
        listener = RedirectListener()
        try:
            self._hit(listener, "code=abc&state=st")
            httpx.get(f"http://127.0.0.1:{listener.port}/favicon.ico",
                      timeout=5.0)
            assert listener.wait("st", timeout=5.0) == "abc"
        finally:
            listener.close()

    def test_an_error_without_state_reports_the_error_not_csrf(self):
        """Ordering bug from the field: checking state first turned the
        server's actual complaint into a misleading CSRF message."""
        listener = RedirectListener()
        try:
            threading.Timer(0.05, self._hit, (
                listener, "error=invalid_client&error_description=unknown"
            )).start()
            with pytest.raises(MCPAuthError, match="invalid_client"):
                listener.wait("st", timeout=5.0)
        finally:
            listener.close()

    def test_a_missing_state_says_so_and_lists_what_arrived(self):
        listener = RedirectListener()
        try:
            threading.Timer(0.05, self._hit, (listener, "code=abc")).start()
            with pytest.raises(MCPAuthError, match="no 'state' at all"):
                listener.wait("st", timeout=5.0)
        finally:
            listener.close()

    def test_timeout_says_what_to_do(self):
        listener = RedirectListener()
        try:
            with pytest.raises(MCPAuthError, match="timed out"):
                listener.wait("st", timeout=0.3)
        finally:
            listener.close()

    def test_port_is_bound_before_the_uri_is_quoted(self):
        """Registration names this URI, so it must already be ours."""
        listener = RedirectListener()
        try:
            assert f":{listener.port}/callback" in listener.redirect_uri
            with httpx.Client(timeout=5.0) as c:
                assert c.get(f"http://127.0.0.1:{listener.port}/callback"
                             "?code=x&state=y").status_code == 200
        finally:
            listener.close()


class TestTokenStore:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "t.json"
        save_token("srv", StoredToken(access_token="a", refresh_token="r"),
                   path)
        loaded = load_token("srv", path)
        assert loaded.access_token == "a" and loaded.refresh_token == "r"

    def test_file_is_not_world_readable(self, tmp_path):
        """A refresh token is a long-lived credential; 0600 or nothing."""
        path = tmp_path / "t.json"
        save_token("srv", StoredToken(access_token="a"), path)
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600, oct(mode)

    def test_one_server_does_not_clobber_another(self, tmp_path):
        path = tmp_path / "t.json"
        save_token("a", StoredToken(access_token="1"), path)
        save_token("b", StoredToken(access_token="2"), path)
        assert load_token("a", path).access_token == "1"
        assert load_token("b", path).access_token == "2"

    def test_forget_removes_only_the_named_one(self, tmp_path):
        path = tmp_path / "t.json"
        save_token("a", StoredToken(access_token="1"), path)
        save_token("b", StoredToken(access_token="2"), path)
        assert forget_token("a", path) is True
        assert load_token("a", path) is None
        assert load_token("b", path) is not None
        assert forget_token("a", path) is False  # idempotent

    def test_corrupt_store_means_log_in_again_not_a_dead_cli(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text("{not json")
        assert load_token("srv", path) is None

    def test_missing_expiry_counts_as_live(self):
        """Servers may omit expires_in; guessing 'expired' would mean a
        pointless refresh before every single call."""
        assert StoredToken(access_token="a", expires_at=None).expired() is False

    def test_expiry_is_early_by_the_skew(self):
        soon = StoredToken(access_token="a", expires_at=time.time() + 5)
        later = StoredToken(access_token="a", expires_at=time.time() + 600)
        assert soon.expired() and not later.expired()


class TestAccessTokenFor:
    def test_absent_is_none_not_an_error(self, tmp_path):
        assert access_token_for("srv", path=tmp_path / "t.json") is None

    def test_live_token_is_returned_untouched(self, tmp_path):
        path = tmp_path / "t.json"
        save_token("srv", StoredToken(access_token="live",
                                      expires_at=time.time() + 999), path)
        assert access_token_for("srv", path=path) == "live"

    def test_expired_token_refreshes_and_persists(self, tmp_path):
        path = tmp_path / "t.json"
        save_token("srv", StoredToken(
            access_token="old", refresh_token="rt-1",
            expires_at=time.time() - 1, client_id="cid",
            token_endpoint="https://idp.example/token",
            authorization_endpoint="https://idp.example/oauth",
            issuer=RESOURCE, resource=RESOURCE), path)
        with idp_client(token_payload={"access_token": "new",
                                       "expires_in": 3600}) as client:
            assert access_token_for("srv", path=path, client=client) == "new"
        # and the refresh token survives a response that omitted it
        assert load_token("srv", path).refresh_token == "rt-1"

    def test_expired_without_a_refresh_token_gives_up_cleanly(self, tmp_path):
        path = tmp_path / "t.json"
        save_token("srv", StoredToken(access_token="old",
                                      expires_at=time.time() - 1), path)
        assert access_token_for("srv", path=path) is None

    def test_force_refreshes_a_token_that_still_looks_live(self, tmp_path):
        """What a 401 means: the server disagrees with our arithmetic."""
        path = tmp_path / "t.json"
        save_token("srv", StoredToken(
            access_token="old", refresh_token="rt-1",
            expires_at=time.time() + 999, client_id="cid",
            token_endpoint="https://idp.example/token",
            authorization_endpoint="https://idp.example/oauth",
            issuer=RESOURCE, resource=RESOURCE), path)
        with idp_client(token_payload={"access_token": "forced"}) as client:
            assert access_token_for("srv", path=path, client=client,
                                    force=True) == "forced"


class TestExchange:
    def test_sends_the_verifier_and_the_resource(self):
        seen: list[httpx.Request] = []
        server = AuthServer(issuer="i", authorization_endpoint="a",
                            token_endpoint="https://idp.example/token")
        with idp_client(seen=seen) as client:
            exchange_code(server, code="c", client_id="cid",
                          redirect_uri="http://127.0.0.1:9/callback",
                          verifier="v", resource=RESOURCE, client=client)
        form = parse_qs(seen[-1].content.decode())
        assert form["code_verifier"] == ["v"]
        assert form["grant_type"] == ["authorization_code"]
        assert form["resource"] == [RESOURCE]

    def test_a_token_response_without_a_token_is_refused(self):
        def handler(request):
            return httpx.Response(200, json={"token_type": "Bearer"})
        server = AuthServer(issuer="i", authorization_endpoint="a",
                            token_endpoint="https://idp.example/token")
        client = httpx.Client(transport=httpx.MockTransport(handler))
        with client:
            with pytest.raises(MCPAuthError, match="no access_token"):
                exchange_code(server, code="c", client_id="cid",
                              redirect_uri="r", verifier="v",
                              resource=RESOURCE, client=client)


class TestLoginEndToEnd:
    """Challenge -> discovery -> registration -> browser -> token on disk."""

    def test_the_whole_walk(self, tmp_path):
        path = tmp_path / "t.json"
        opened: list[str] = []

        def fake_browser(url: str) -> None:
            """Stand in for a human: read the URL, answer the redirect."""
            opened.append(url)
            q = parse_qs(urlparse(url).query)
            redirect = q["redirect_uri"][0]
            state = q["state"][0]
            threading.Timer(0.05, lambda: httpx.get(
                f"{redirect}?code=the-code&state={state}", timeout=5.0)
            ).start()

        seen: list[httpx.Request] = []
        with idp_client(seen=seen) as client:
            token = login("srv", RESOURCE, CHALLENGE,
                          open_browser=fake_browser, client=client,
                          path=path, timeout=10.0)

        assert token.access_token == "at-1"
        assert token.refresh_token == "rt-1"
        assert load_token("srv", path).access_token == "at-1"

        # the port named at registration is the port the browser came back
        # to -- the bug that a rebound listener would have introduced
        registered = json.loads(
            next(r for r in seen if r.url.path.endswith("/register")).content
        )["redirect_uris"][0]
        assert parse_qs(urlparse(opened[0]).query)["redirect_uri"] == [
            registered]

    def test_on_url_lets_a_headless_box_copy_the_link(self, tmp_path):
        shown: list[str] = []

        def answer(url: str) -> None:
            q = parse_qs(urlparse(url).query)
            threading.Timer(0.05, lambda: httpx.get(
                f"{q['redirect_uri'][0]}?code=c&state={q['state'][0]}",
                timeout=5.0)).start()

        with idp_client() as client:
            login("srv", RESOURCE, CHALLENGE, open_browser=answer,
                  on_url=shown.append, client=client,
                  path=tmp_path / "t.json", timeout=10.0)
        assert shown and shown[0].startswith("https://idp.example/oauth?")

    def test_a_refused_login_saves_nothing(self, tmp_path):
        path = tmp_path / "t.json"

        def refuse(url: str) -> None:
            q = parse_qs(urlparse(url).query)
            threading.Timer(0.05, lambda: httpx.get(
                f"{q['redirect_uri'][0]}?error=access_denied"
                f"&state={q['state'][0]}", timeout=5.0)).start()

        with idp_client() as client:
            with pytest.raises(MCPAuthError, match="access_denied"):
                login("srv", RESOURCE, CHALLENGE, open_browser=refuse,
                      client=client, path=path, timeout=10.0)
        assert not path.exists()
