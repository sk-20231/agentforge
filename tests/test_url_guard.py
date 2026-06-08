"""Unit tests for the SSRF URL guard (agentforge.safety.is_safe_url, Step 17d).

These use numeric IP literals (and obviously-bad hosts) so they don't depend on
real DNS / network: getaddrinfo on a numeric IP returns it without a lookup.
"""
from agentforge.safety import is_safe_url


class TestBlocked:
    def test_loopback_name(self):
        ok, reason = is_safe_url("http://localhost/admin")
        assert ok is False and "non-public" in reason or "resolve" in reason

    def test_loopback_ip(self):
        assert is_safe_url("http://127.0.0.1:8080/")[0] is False

    def test_cloud_metadata(self):
        # The classic SSRF target — link-local, hands out cloud credentials.
        assert is_safe_url("http://169.254.169.254/latest/meta-data/")[0] is False

    def test_private_10(self):
        assert is_safe_url("http://10.0.0.5/")[0] is False

    def test_private_192(self):
        assert is_safe_url("https://192.168.1.1/")[0] is False

    def test_unspecified(self):
        assert is_safe_url("http://0.0.0.0/")[0] is False

    def test_ipv6_loopback(self):
        assert is_safe_url("http://[::1]/")[0] is False

    def test_non_http_scheme(self):
        ok, reason = is_safe_url("file:///etc/passwd")
        assert ok is False and "scheme" in reason

    def test_no_scheme(self):
        assert is_safe_url("example.com/path")[0] is False

    def test_empty(self):
        assert is_safe_url("")[0] is False


class TestAllowed:
    def test_public_ip(self):
        ok, reason = is_safe_url("http://8.8.8.8/page")
        assert ok is True and reason == "ok"

    def test_public_ip_https(self):
        assert is_safe_url("https://1.1.1.1/")[0] is True
