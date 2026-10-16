# -*- coding: utf-8 -*-

"""
Unit tests for kiro/client_profiles.py.

Expected values are pinned to real captures (2026-10-16) of kiro-cli 2.2.2 and
Kiro IDE 1.3.11. If an official client changes its wire format, update the
profile AND these fixtures together after a fresh capture.
"""

import dataclasses
import uuid

import pytest

import kiro.client_profiles as cp
from kiro.client_profiles import (
    CLI_PROFILE,
    CONTENT_TYPE_AWS_JSON_1_0,
    IDE_PROFILE,
    PROFILES,
    ClientProfileName,
    CredentialSource,
    KiroOperation,
    _node_os_name,
    _rust_os_name,
    detect_profile_for_credentials,
    parse_profile_name,
    resolve_client_profile,
)

ARN = "arn:aws:codewhisperer:us-east-1:123456789012:profile/EXAMPLEPROFILE"
FP = "0000000000000000000000000000000000000000000000000000000000000000"
GAR = KiroOperation.GENERATE_ASSISTANT_RESPONSE
LAM = KiroOperation.LIST_AVAILABLE_MODELS
LAP = KiroOperation.LIST_AVAILABLE_PROFILES
ALL_OPERATIONS = list(KiroOperation)
# ListAvailableProfiles was verified live, not TLS-captured, so it is checked separately
CAPTURED_OPERATIONS = [GAR, LAM]

# Captured header sets (authorization, invocation id, host, length, connection removed).
# OS-dependent parts use the Linux machine the capture was taken on.
CAPTURED_CLI = {
    GAR: {
        "content-type": "application/x-amz-json-1.0",
        "x-amz-target": "AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
        "user-agent": "aws-sdk-rust/1.3.14 ua/2.1 api/codewhispererstreaming/0.1.14474 os/linux lang/rust/1.92.0 md/appVersion-2.2.2 app/AmazonQ-For-CLI",
        "x-amz-user-agent": "aws-sdk-rust/1.3.14 ua/2.1 api/codewhispererstreaming/0.1.14474 os/linux lang/rust/1.92.0 m/F app/AmazonQ-For-CLI",
        "amz-sdk-request": "attempt=1; max=3",
        "accept": "*/*",
        "accept-encoding": "gzip",
    },
    LAM: {
        "content-type": "application/x-amz-json-1.0",
        "x-amz-target": "AmazonCodeWhispererService.ListAvailableModels",
        "user-agent": "aws-sdk-rust/1.3.14 ua/2.1 api/codewhispererruntime/0.1.14474 os/linux lang/rust/1.92.0 md/appVersion-2.2.2 app/AmazonQ-For-CLI",
        "x-amz-user-agent": "aws-sdk-rust/1.3.14 ua/2.1 api/codewhispererruntime/0.1.14474 os/linux lang/rust/1.92.0 m/F,C app/AmazonQ-For-CLI",
        "amz-sdk-request": "attempt=1; max=3",
        "accept": "*/*",
        "accept-encoding": "gzip",
    },
}
IDE_TAG = f"KiroIDE-1.3.11-{FP}-KAS/0.66.34"
IDE_OS = "linux#6.8.0-138-generic"
CAPTURED_IDE = {
    GAR: {
        "content-type": "application/x-amz-json-1.0",
        "x-amz-target": "KiroRuntimeService.GenerateAssistantResponse",
        "x-amzn-codewhisperer-optout": "true",
        "tokentype": "SSO_OIDC",
        "x-amzn-kiro-client-attribution": "kiro-ide",
        "x-amz-user-agent": f"aws-sdk-js/1.0.0 {IDE_TAG}",
        "user-agent": f"aws-sdk-js/1.0.0 ua/2.1 os/{IDE_OS} lang/js md/nodejs#24.21.0 api/kiroruntime#1.0.0 m/N,E {IDE_TAG}",
        "amz-sdk-request": "attempt=1; max=1",
        "x-kiro-attempt": "1;max=1",
    },
    LAM: {
        "content-type": "application/x-amz-json-1.0",
        "x-amz-target": "KiroControlPlaneBearerService.ListAvailableModels",
        "tokentype": "SSO_OIDC",
        "x-amz-user-agent": f"aws-sdk-js/1.0.0 {IDE_TAG}",
        "user-agent": f"aws-sdk-js/1.0.0 ua/2.1 os/{IDE_OS} lang/js md/nodejs#24.21.0 api/kirocontrolplanebearer#1.0.0 m/N,E {IDE_TAG}",
        "amz-sdk-request": "attempt=1; max=3",
    },
}
VOLATILE = {"authorization", "amz-sdk-invocation-id"}


@pytest.fixture
def capture_machine(monkeypatch):
    """Pin OS strings to the Linux machine the captures were taken on."""
    monkeypatch.setattr(cp, "CLI_OS", "linux")
    monkeypatch.setattr(cp, "IDE_OS", IDE_OS)


def _normalized(headers):
    return {k.lower(): v for k, v in headers.items() if k.lower() not in VOLATILE}


class TestMatchesCapturedTraffic:
    """Generated requests must equal the captured ones (host, path, headers)."""

    @pytest.mark.parametrize("operation", CAPTURED_OPERATIONS)
    def test_cli_headers_match_capture_except_optout(self, capture_machine, operation):
        """
        What it does: Compares every CLI header with the kiro-cli 2.2.2 capture.
        Purpose: Guarantee CLI fingerprint fidelity; optout is the only documented deviation.
        """
        generated = _normalized(CLI_PROFILE.build_headers("tok", "unused", operation))
        assert generated.pop("x-amzn-codewhisperer-optout") == "true"
        assert generated == CAPTURED_CLI[operation]

    @pytest.mark.parametrize("operation", CAPTURED_OPERATIONS)
    def test_ide_headers_match_capture_for_sso_oidc(self, capture_machine, operation):
        """
        What it does: Compares every IDE header with the Kiro IDE 1.3.11 capture (IdC login).
        Purpose: Guarantee IDE fingerprint fidelity with no deviations.
        """
        generated = _normalized(IDE_PROFILE.build_headers("tok", FP, operation, is_sso_oidc=True))
        assert generated == CAPTURED_IDE[operation]

    @pytest.mark.parametrize("operation", CAPTURED_OPERATIONS)
    def test_ide_omits_token_type_for_non_oidc(self, capture_machine, operation):
        """
        What it does: Builds IDE headers for a Kiro Desktop (social) token.
        Purpose: TokenType value for social tokens is unverified, so it is not sent.
        """
        generated = _normalized(IDE_PROFILE.build_headers("tok", FP, operation, is_sso_oidc=False))
        expected = {k: v for k, v in CAPTURED_IDE[operation].items() if k != "tokentype"}
        assert generated == expected

    @pytest.mark.parametrize("operation", ALL_OPERATIONS)
    @pytest.mark.parametrize("is_sso_oidc", [True, False])
    def test_cli_never_sends_token_type(self, operation, is_sso_oidc):
        """
        What it does: Builds CLI headers for both token kinds.
        Purpose: kiro-cli never sends the IDE-only TokenType header.
        """
        headers = CLI_PROFILE.build_headers("tok", "fp", operation, is_sso_oidc=is_sso_oidc)
        assert "TokenType" not in headers

    @pytest.mark.parametrize("profile, operation, url", [
        (CLI_PROFILE, GAR, "https://q.us-east-1.amazonaws.com/"),
        (CLI_PROFILE, LAM, "https://q.us-east-1.amazonaws.com/"),
        (IDE_PROFILE, GAR, "https://runtime.us-east-1.kiro.dev/"),
        (IDE_PROFILE, LAM, "https://management.us-east-1.kiro.dev/"),
    ])
    def test_urls_match_capture(self, profile, operation, url):
        """
        What it does: Builds every operation URL for us-east-1.
        Purpose: Host and path must equal what each official client calls.
        """
        assert profile.build_url("us-east-1", operation) == url

    @pytest.mark.parametrize("region", ["eu-central-1", "us-gov-west-1"])
    def test_urls_follow_region(self, region):
        """
        What it does: Builds URLs for other regions.
        Purpose: Region must be substituted in every host template.
        """
        assert CLI_PROFILE.build_url(region, GAR) == f"https://q.{region}.amazonaws.com/"
        assert IDE_PROFILE.build_url(region, GAR) == f"https://runtime.{region}.kiro.dev/"
        assert IDE_PROFILE.build_url(region, LAM) == f"https://management.{region}.kiro.dev/"

    def test_cli_list_models_args_in_query_and_body(self):
        """
        What it does: Builds CLI ListAvailableModels.
        Purpose: kiro-cli sends origin/profileArn in BOTH query string and JSON body.
        """
        spec = CLI_PROFILE.build_list_models_request("us-east-1", ARN)
        assert (spec.method, spec.url) == ("POST", "https://q.us-east-1.amazonaws.com/")
        assert spec.params == {"origin": "KIRO_CLI", "profileArn": ARN}
        assert spec.json_data == {"origin": "KIRO_CLI", "profileArn": ARN}
        assert spec.params is not spec.json_data

    def test_ide_list_models_args_in_body_only(self):
        """
        What it does: Builds IDE ListAvailableModels.
        Purpose: Kiro IDE sends arguments only in the JSON body to the management host.
        """
        spec = IDE_PROFILE.build_list_models_request("us-east-1", ARN)
        assert (spec.method, spec.url) == ("POST", "https://management.us-east-1.kiro.dev/")
        assert spec.params is None
        assert spec.json_data == {"origin": "AI_EDITOR", "profileArn": ARN}

    @pytest.mark.parametrize("profile", [CLI_PROFILE, IDE_PROFILE])
    @pytest.mark.parametrize("arn", [None, ""])
    def test_list_models_without_arn(self, profile, arn):
        """
        What it does: Builds ListAvailableModels without a profile ARN.
        Purpose: Never send an empty/None profileArn.
        """
        spec = profile.build_list_models_request("us-east-1", arn)
        assert spec.json_data == {"origin": profile.origin}
        if spec.params is not None:
            assert spec.params == {"origin": profile.origin}


class TestOsStrings:
    """Tests for OS name mapping used in user agents."""

    @pytest.mark.parametrize("system, expected", [
        ("Linux", "linux"), ("Darwin", "macos"), ("Windows", "windows"), ("FreeBSD", "freebsd"), ("", "other"),
    ])
    def test_rust_os(self, system, expected):
        """
        What it does: Maps platform.system() to aws-sdk-rust values.
        Purpose: Match std::env::consts::OS naming used by kiro-cli.
        """
        assert _rust_os_name(system) == expected

    @pytest.mark.parametrize("system, release, expected", [
        ("Linux", "6.8.0-138-generic", "linux#6.8.0-138-generic"),
        ("Darwin", "24.1.0", "darwin#24.1.0"),
        ("Windows", "10", "win32#10"),
        ("Linux", "", "linux"),
    ])
    def test_node_os(self, system, release, expected):
        """
        What it does: Maps platform info to aws-sdk-js values.
        Purpose: Match process.platform#os.release() naming used by Kiro IDE.
        """
        assert _node_os_name(system, release) == expected


class TestHeaderInvariants:
    """Structural invariants for every profile and operation."""

    @pytest.mark.parametrize("profile", [CLI_PROFILE, IDE_PROFILE])
    @pytest.mark.parametrize("operation", ALL_OPERATIONS)
    def test_rpc_protocol_and_sdk_family(self, profile, operation):
        """
        What it does: Checks protocol headers and SDK family per profile.
        Purpose: No hybrid fingerprints (e.g. rust UA with IDE target).
        """
        headers = profile.build_headers("tok", "fp", operation)
        assert headers["Content-Type"] == CONTENT_TYPE_AWS_JSON_1_0
        assert headers["x-amz-target"] == profile.operation(operation).target
        family = "aws-sdk-rust/" if profile is CLI_PROFILE else "aws-sdk-js/"
        assert headers["User-Agent"].startswith(family)
        assert headers["x-amz-user-agent"].startswith(family)
        is_kiro_service = headers["x-amz-target"].startswith("Kiro")
        assert is_kiro_service is (profile is IDE_PROFILE)

    def test_cli_user_agent_does_not_leak_fingerprint(self):
        """
        What it does: Checks CLI headers for the machine fingerprint.
        Purpose: kiro-cli never sends one.
        """
        headers = CLI_PROFILE.build_headers("tok", "UNIQUE_FP_MARKER", GAR)
        assert all("UNIQUE_FP_MARKER" not in v for v in headers.values())

    def test_cli_user_agent_has_no_exec_env(self):
        """
        What it does: Checks CLI user agent for exec-env.
        Purpose: exec-env only appears when AWS_EXECUTION_ENV is set in kiro-cli's
        environment (e.g. inside kiro-cli chat tools); a normal kiro-cli does not send it.
        """
        assert "exec-env" not in CLI_PROFILE.build_headers("tok", "fp", GAR)["User-Agent"]

    @pytest.mark.parametrize("profile", [CLI_PROFILE, IDE_PROFILE])
    def test_invocation_id_unique_and_dict_fresh(self, profile):
        """
        What it does: Builds headers twice and mutates the first result.
        Purpose: Fresh invocation ids; callers mutate headers (Connection: close).
        """
        first = profile.build_headers("tok", "fp", GAR)
        first["Connection"] = "close"
        second = profile.build_headers("tok", "fp", GAR)
        uuid.UUID(first["amz-sdk-invocation-id"])
        assert first["amz-sdk-invocation-id"] != second["amz-sdk-invocation-id"]
        assert "Connection" not in second

    @pytest.mark.parametrize("token", ["", "a" * 5000, "tok with spaces"])
    def test_unusual_tokens_passed_verbatim(self, token):
        """
        What it does: Passes empty, huge and spaced tokens.
        Purpose: Header builder must not transform tokens.
        """
        assert CLI_PROFILE.build_headers(token, "fp", GAR)["Authorization"] == f"Bearer {token}"

    @pytest.mark.parametrize("profile", [CLI_PROFILE, IDE_PROFILE])
    def test_optout_always_true(self, profile):
        """
        What it does: Checks the opt-out header on chat requests.
        Purpose: Gateway policy is to opt out of data usage for every profile.
        """
        assert profile.build_headers("tok", "fp", GAR)["x-amzn-codewhisperer-optout"] == "true"


class TestApplyPayloadOrigin:
    """Tests for ClientProfile.apply_payload_origin()."""

    @staticmethod
    def _payload():
        return {
            "conversationState": {
                "currentMessage": {"userInputMessage": {"content": "hi", "origin": "AI_EDITOR"}},
                "history": [
                    {"userInputMessage": {"content": "a", "origin": "AI_EDITOR"}},
                    {"assistantResponseMessage": {"content": "b"}},
                    {"userInputMessage": {"content": "c", "origin": "AI_EDITOR"}},
                ],
            },
            "profileArn": ARN,
        }

    @pytest.mark.parametrize("profile, origin", [(CLI_PROFILE, "KIRO_CLI"), (IDE_PROFILE, "AI_EDITOR")])
    def test_sets_origin_everywhere(self, profile, origin):
        """
        What it does: Applies origin to current message and history.
        Purpose: kiro-cli sends KIRO_CLI, IDE sends AI_EDITOR.
        """
        payload = profile.apply_payload_origin(self._payload())
        state = payload["conversationState"]
        assert state["currentMessage"]["userInputMessage"]["origin"] == origin
        assert [m["userInputMessage"]["origin"] for m in state["history"] if "userInputMessage" in m] == [origin, origin]

    def test_content_untouched(self):
        """
        What it does: Compares payload before/after except origin.
        Purpose: Gateway never modifies user content.
        """
        before = self._payload()
        after = CLI_PROFILE.apply_payload_origin(self._payload())
        for msg in (after["conversationState"]["currentMessage"], *after["conversationState"]["history"]):
            msg.get("userInputMessage", {}).pop("origin", None)
        for msg in (before["conversationState"]["currentMessage"], *before["conversationState"]["history"]):
            msg.get("userInputMessage", {}).pop("origin", None)
        assert after == before

    def test_returns_same_object(self):
        """
        What it does: Checks in-place behavior.
        Purpose: Callers rely on mutation, no copy.
        """
        payload = self._payload()
        assert CLI_PROFILE.apply_payload_origin(payload) is payload

    @pytest.mark.parametrize("payload", [
        {},
        {"conversationState": None},
        {"conversationState": "x"},
        {"conversationState": {"currentMessage": None, "history": None}},
        {"conversationState": {"currentMessage": {"userInputMessage": None}, "history": ["junk", 1, None]}},
        {"conversationState": {"currentMessage": {"userInputMessage": {"content": "no origin"}}}},
    ])
    def test_malformed_payloads_do_not_crash(self, payload):
        """
        What it does: Applies origin to malformed/partial payloads.
        Purpose: Must be robust and never add missing origin keys.
        """
        import copy
        original = copy.deepcopy(payload)
        assert CLI_PROFILE.apply_payload_origin(payload) == original


class TestParseProfileName:
    """Tests for parse_profile_name()."""

    @pytest.mark.parametrize("raw, expected", [
        ("auto", ClientProfileName.AUTO), ("cli", ClientProfileName.CLI), ("ide", ClientProfileName.IDE),
        ("CLI", ClientProfileName.CLI), ("  Ide  ", ClientProfileName.IDE),
        ("", ClientProfileName.AUTO), ("   ", ClientProfileName.AUTO), (None, ClientProfileName.AUTO),
    ])
    def test_valid_values_parse(self, raw, expected):
        """
        What it does: Parses valid, mixed-case, padded and empty values.
        Purpose: Forgiving config; empty means auto.
        """
        assert parse_profile_name(raw) == expected

    @pytest.mark.parametrize("raw", ["desktop", "kiro-cli", "c l i", "auto,cli", "0", "idee"])
    def test_invalid_values_raise_actionable_error(self, raw):
        """
        What it does: Rejects unknown profile names.
        Purpose: Typos fail loudly with allowed values listed.
        """
        with pytest.raises(ValueError) as exc_info:
            parse_profile_name(raw)
        message = str(exc_info.value)
        assert raw in message and "auto, cli, ide" in message and "KIRO_CLIENT_PROFILE" in message


class TestDetectAndResolve:
    """Tests for detect_profile_for_credentials() and resolve_client_profile()."""

    @pytest.mark.parametrize("source, is_ide_file, expected", [
        (CredentialSource.SQLITE, True, ClientProfileName.CLI),
        (CredentialSource.SQLITE, False, ClientProfileName.CLI),
        (CredentialSource.JSON_FILE, True, ClientProfileName.IDE),
        (CredentialSource.JSON_FILE, False, ClientProfileName.CLI),
        (CredentialSource.REFRESH_TOKEN, True, ClientProfileName.IDE),
        (CredentialSource.REFRESH_TOKEN, False, ClientProfileName.IDE),
    ])
    def test_detection_matrix(self, source, is_ide_file, expected):
        """
        What it does: Evaluates every auto-detection input combination.
        Purpose: Auto picks the client that produced the credentials, never AUTO.
        """
        assert detect_profile_for_credentials(source, is_ide_file) == expected

    @pytest.mark.parametrize("requested, expected", [("cli", CLI_PROFILE), ("ide", IDE_PROFILE), ("IDE", IDE_PROFILE)])
    @pytest.mark.parametrize("source", list(CredentialSource))
    def test_explicit_profile_overrides_source(self, requested, expected, source):
        """
        What it does: Explicit profile wins over credential source.
        Purpose: Users can force a fingerprint.
        """
        assert resolve_client_profile(requested, source, True) is expected
        assert resolve_client_profile(requested, source, False) is expected

    @pytest.mark.parametrize("requested", ["auto", "", None])
    def test_auto_follows_source(self, requested):
        """
        What it does: auto/empty/None delegate to detection.
        Purpose: Default behavior matches credential origin.
        """
        assert resolve_client_profile(requested, CredentialSource.SQLITE, False) is CLI_PROFILE
        assert resolve_client_profile(requested, CredentialSource.JSON_FILE, True) is IDE_PROFILE

    def test_invalid_raises(self):
        """
        What it does: Propagates validation errors.
        Purpose: Invalid values never silently fall back.
        """
        with pytest.raises(ValueError):
            resolve_client_profile("bogus", CredentialSource.SQLITE, False)


class TestProfileRegistry:
    """Tests for the PROFILES registry integrity."""

    def test_registry_has_only_concrete_profiles(self):
        """
        What it does: Inspects registry keys.
        Purpose: AUTO is a selector, not a profile.
        """
        assert set(PROFILES) == {ClientProfileName.CLI, ClientProfileName.IDE}
        assert all(PROFILES[name].name == name for name in PROFILES)

    @pytest.mark.parametrize("profile", [CLI_PROFILE, IDE_PROFILE])
    def test_every_operation_defined(self, profile):
        """
        What it does: Checks each profile defines every operation with a valid host.
        Purpose: Adding an operation without wiring all profiles must fail tests.
        """
        for operation in ALL_OPERATIONS:
            spec = profile.operation(operation)
            assert spec.path.startswith("/")
            assert spec.host_template.startswith("https://") and "{region}" in spec.host_template

    def test_profiles_are_immutable(self):
        """
        What it does: Attempts to mutate a profile.
        Purpose: Shared singletons must stay frozen.
        """
        with pytest.raises(dataclasses.FrozenInstanceError):
            CLI_PROFILE.origin = "X"  # type: ignore[misc]


class TestListAvailableProfilesWireFormat:
    """ListAvailableProfiles host and target, as verified live on 2026-10-16."""

    @pytest.mark.parametrize("profile, url, target", [
        (CLI_PROFILE, "https://q.us-east-1.amazonaws.com/", "AmazonCodeWhispererService.ListAvailableProfiles"),
        (IDE_PROFILE, "https://management.us-east-1.kiro.dev/", "KiroControlPlaneBearerService.ListAvailableProfiles"),
    ])
    def test_url_and_target(self, profile, url, target):
        """
        What it does: Verifies the host and x-amz-target of ListAvailableProfiles.
        Purpose: These are the only combinations that answered with HTTP 200.
        """
        assert profile.build_url("us-east-1", LAP) == url
        headers = profile.build_headers("tok", FP, LAP, is_sso_oidc=True)
        assert headers["x-amz-target"] == target
        assert headers["Authorization"] == "Bearer tok"
