# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Client fingerprint profiles for Kiro API requests.

Each profile reproduces the wire format of one official client, taken from real
TLS captures (authorization redacted) on 2026-10-16:

kiro-cli 2.2.2 (aws-sdk-rust, AWS JSON 1.0 RPC):
    GenerateAssistantResponse  POST https://q.{region}.amazonaws.com/
        x-amz-target: AmazonCodeWhispererStreamingService.GenerateAssistantResponse
    ListAvailableModels        POST https://q.{region}.amazonaws.com/?origin=KIRO_CLI&profileArn=...
        x-amz-target: AmazonCodeWhispererService.ListAvailableModels
        (arguments sent both in the query string and in the JSON body)
    userInputMessage.origin = KIRO_CLI

Kiro IDE 1.3.11 (aws-sdk-js, AWS JSON 1.0 RPC):
    GenerateAssistantResponse  POST https://runtime.{region}.kiro.dev/
        x-amz-target: KiroRuntimeService.GenerateAssistantResponse
    ListAvailableModels        POST https://management.{region}.kiro.dev/
        x-amz-target: KiroControlPlaneBearerService.ListAvailableModels
    userInputMessage.origin = AI_EDITOR

A profile bundles host, path, method, content type, operation target, user agents
and client-specific headers, so a request never mixes traits of two clients.

Deliberate deviations from the captures (documented, not routing-relevant):
- x-amzn-codewhisperer-optout is always "true" (kiro-cli sends "false"); this is a
  data-sharing choice made by the gateway, not a client trait.
- The IDE "TokenType: SSO_OIDC" header was only observed for AWS SSO OIDC logins;
  it is omitted for Kiro Desktop (social) tokens because their value is unverified.
- The machine id in the IDE user agent is the gateway's own fingerprint (same
  64-hex format), not the IDE's machine id.

Profile selection (KIRO_CLIENT_PROFILE or per-account "client_profile"):
- "cli": always use the kiro-cli wire format
- "ide": always use the Kiro IDE wire format
- "auto" (default): follow where the credentials came from
    (kiro-cli SQLite / raw AWS SSO cache -> cli, Kiro IDE files / refresh token -> ide)
"""

import platform
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional

from loguru import logger


class ClientProfileName(str, Enum):
    """Configurable client profile names."""

    AUTO = "auto"
    CLI = "cli"
    IDE = "ide"


class KiroOperation(str, Enum):
    """Kiro API operations whose wire format depends on the client profile."""

    GENERATE_ASSISTANT_RESPONSE = "GenerateAssistantResponse"
    LIST_AVAILABLE_MODELS = "ListAvailableModels"


class CredentialSource(str, Enum):
    """Where an account's credentials were loaded from (input for "auto" selection)."""

    SQLITE = "sqlite"
    JSON_FILE = "json_file"
    REFRESH_TOKEN = "refresh_token"


CONTENT_TYPE_AWS_JSON_1_0: str = "application/x-amz-json-1.0"

# Gateway policy: always opt out of service-improvement data usage (see module docstring)
OPTOUT_HEADER_VALUE: str = "true"

# Value of the IDE "TokenType" header for AWS SSO OIDC tokens (captured)
TOKEN_TYPE_SSO_OIDC: str = "SSO_OIDC"

# --- Hosts (captured) ---
CLI_HOST_TEMPLATE: str = "https://q.{region}.amazonaws.com"
IDE_RUNTIME_HOST_TEMPLATE: str = "https://runtime.{region}.kiro.dev"
IDE_MANAGEMENT_HOST_TEMPLATE: str = "https://management.{region}.kiro.dev"

# --- kiro-cli 2.2.2 fingerprint (aws-sdk-rust) ---
CLI_APP_VERSION: str = "2.2.2"
CLI_SDK: str = "aws-sdk-rust/1.3.14 ua/2.1"
CLI_API_VERSION: str = "0.1.14474"
CLI_LANG: str = "lang/rust/1.92.0"
CLI_ORIGIN: str = "KIRO_CLI"

# --- Kiro IDE 1.3.11 fingerprint (aws-sdk-js) ---
IDE_APP_VERSION: str = "1.3.11"
IDE_SDK: str = "aws-sdk-js/1.0.0"
IDE_KAS_VERSION: str = "0.66.34"
IDE_NODE_VERSION: str = "24.21.0"
IDE_ORIGIN: str = "AI_EDITOR"


def _rust_os_name(system: str) -> str:
    """
    Map platform.system() to the aws-sdk-rust "os/" value (std::env::consts::OS).

    Args:
        system: Value of platform.system() (e.g. "Linux", "Darwin", "Windows")

    Returns:
        "linux", "macos", "windows" or the lowercased system name
    """
    return {"linux": "linux", "darwin": "macos", "windows": "windows"}.get(system.lower(), system.lower() or "other")


def _node_os_name(system: str, release: str) -> str:
    """
    Map platform info to the aws-sdk-js "os/" value (process.platform#os.release()).

    Args:
        system: Value of platform.system()
        release: Value of platform.release()

    Returns:
        e.g. "linux#6.8.0-138-generic", "darwin#24.1.0", "win32#10.0.19044"
    """
    name = {"linux": "linux", "darwin": "darwin", "windows": "win32"}.get(system.lower(), system.lower() or "other")
    return f"{name}#{release}" if release else name


CLI_OS: str = _rust_os_name(platform.system())
IDE_OS: str = _node_os_name(platform.system(), platform.release())


@dataclass(frozen=True)
class OperationSpec:
    """
    Wire format of one operation for one client profile.

    Attributes:
        host_template: Host URL template with "{region}"
        path: URL path appended to the host
        target: x-amz-target header value
        sdk_api: SDK API identifier used in the user agent
        sdk_max_attempts: "max" value of the amz-sdk-request header
        cli_metrics: aws-sdk-rust x-amz-user-agent metrics ("m/...") - CLI only
        args_in_query: Also send operation arguments in the query string
        extra_headers: Operation-specific static headers
    """

    host_template: str
    path: str
    target: str
    sdk_api: str
    sdk_max_attempts: int
    cli_metrics: str = ""
    args_in_query: bool = False
    extra_headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KiroRequestSpec:
    """
    Fully resolved non-header parts of a Kiro request.

    Attributes:
        method: HTTP method
        url: Absolute URL
        params: Query parameters (None if not used)
        json_data: JSON body (None if not used)
    """

    method: str
    url: str
    params: Optional[Dict[str, str]]
    json_data: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class ClientProfile:
    """
    Complete, self-consistent fingerprint of one official Kiro client.

    Attributes:
        name: Profile name (CLI or IDE, never AUTO)
        operations: Wire format per operation
        origin: "origin" value for chat payloads and ListAvailableModels
    """

    name: ClientProfileName
    operations: Mapping[KiroOperation, OperationSpec]
    origin: str

    def operation(self, operation: KiroOperation) -> OperationSpec:
        """
        Return the wire format for an operation.

        Args:
            operation: Kiro operation

        Returns:
            OperationSpec for this profile

        Raises:
            KeyError: If the profile does not define the operation (programming error)
        """
        return self.operations[operation]

    def build_url(self, region: str, operation: KiroOperation) -> str:
        """
        Build the absolute URL for an operation.

        Args:
            region: API region (e.g. "us-east-1")
            operation: Kiro operation

        Returns:
            Absolute URL, e.g. "https://q.us-east-1.amazonaws.com/"
        """
        spec = self.operation(operation)
        return spec.host_template.format(region=region) + spec.path

    def user_agents(self, operation: KiroOperation, fingerprint: str) -> Dict[str, str]:
        """
        Build User-Agent and x-amz-user-agent headers.

        Args:
            operation: Kiro operation (user agents embed the SDK API name)
            fingerprint: Machine fingerprint (IDE only)

        Returns:
            Dict with "User-Agent" and "x-amz-user-agent"
        """
        spec = self.operation(operation)
        if self.name == ClientProfileName.CLI:
            base = f"{CLI_SDK} api/{spec.sdk_api}/{CLI_API_VERSION} os/{CLI_OS} {CLI_LANG}"
            return {
                "User-Agent": f"{base} md/appVersion-{CLI_APP_VERSION} app/AmazonQ-For-CLI",
                "x-amz-user-agent": f"{base} {spec.cli_metrics} app/AmazonQ-For-CLI",
            }
        ide_tag = f"KiroIDE-{IDE_APP_VERSION}-{fingerprint}-KAS/{IDE_KAS_VERSION}"
        return {
            "User-Agent": (
                f"{IDE_SDK} ua/2.1 os/{IDE_OS} lang/js md/nodejs#{IDE_NODE_VERSION} "
                f"api/{spec.sdk_api}#1.0.0 m/N,E {ide_tag}"
            ),
            "x-amz-user-agent": f"{IDE_SDK} {ide_tag}",
        }

    def build_headers(
        self,
        token: str,
        fingerprint: str,
        operation: KiroOperation,
        is_sso_oidc: bool = False,
    ) -> Dict[str, str]:
        """
        Build all request headers for an operation.

        Args:
            token: Bearer access token
            fingerprint: Machine fingerprint
            operation: Kiro operation
            is_sso_oidc: Whether the token is an AWS SSO OIDC token (adds IDE TokenType)

        Returns:
            Header dict (a fresh dict; callers may mutate it)
        """
        spec = self.operation(operation)
        headers: Dict[str, str] = {
            "Content-Type": CONTENT_TYPE_AWS_JSON_1_0,
            "x-amz-target": spec.target,
            "Authorization": f"Bearer {token}",
        }
        headers.update(self.user_agents(operation, fingerprint))
        headers["amz-sdk-request"] = f"attempt=1; max={spec.sdk_max_attempts}"
        headers["amz-sdk-invocation-id"] = str(uuid.uuid4())
        if self.name == ClientProfileName.CLI:
            headers["x-amzn-codewhisperer-optout"] = OPTOUT_HEADER_VALUE
            headers["accept"] = "*/*"
            headers["accept-encoding"] = "gzip"
        elif is_sso_oidc:
            headers["TokenType"] = TOKEN_TYPE_SSO_OIDC
        headers.update(spec.extra_headers)
        return headers

    def build_list_models_request(self, region: str, profile_arn: Optional[str]) -> KiroRequestSpec:
        """
        Build the ListAvailableModels request in this profile's wire format.

        Args:
            region: API region
            profile_arn: Profile ARN, sent only when known

        Returns:
            KiroRequestSpec for KiroHttpClient.request_with_retry
        """
        spec = self.operation(KiroOperation.LIST_AVAILABLE_MODELS)
        args: Dict[str, str] = {"origin": self.origin}
        if profile_arn:
            args["profileArn"] = profile_arn
        return KiroRequestSpec(
            method="POST",
            url=self.build_url(region, KiroOperation.LIST_AVAILABLE_MODELS),
            params=dict(args) if spec.args_in_query else None,
            json_data=args,
        )

    def apply_payload_origin(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set this client's origin on every userInputMessage of a Kiro chat payload.

        Only existing "origin" keys are rewritten; content is never touched.

        Args:
            payload: Kiro generateAssistantResponse payload (modified in place)

        Returns:
            The same payload object
        """
        state = payload.get("conversationState")
        if not isinstance(state, dict):
            return payload
        messages = [state.get("currentMessage")]
        history = state.get("history")
        if isinstance(history, list):
            messages.extend(history)
        for message in messages:
            if not isinstance(message, dict):
                continue
            user_input = message.get("userInputMessage")
            if isinstance(user_input, dict) and "origin" in user_input:
                user_input["origin"] = self.origin
        return payload


CLI_PROFILE = ClientProfile(
    name=ClientProfileName.CLI,
    operations={
        KiroOperation.GENERATE_ASSISTANT_RESPONSE: OperationSpec(
            host_template=CLI_HOST_TEMPLATE,
            path="/",
            target="AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
            sdk_api="codewhispererstreaming",
            sdk_max_attempts=3,
            cli_metrics="m/F",
        ),
        KiroOperation.LIST_AVAILABLE_MODELS: OperationSpec(
            host_template=CLI_HOST_TEMPLATE,
            path="/",
            target="AmazonCodeWhispererService.ListAvailableModels",
            sdk_api="codewhispererruntime",
            sdk_max_attempts=3,
            cli_metrics="m/F,C",
            args_in_query=True,
        ),
    },
    origin=CLI_ORIGIN,
)

IDE_PROFILE = ClientProfile(
    name=ClientProfileName.IDE,
    operations={
        KiroOperation.GENERATE_ASSISTANT_RESPONSE: OperationSpec(
            host_template=IDE_RUNTIME_HOST_TEMPLATE,
            path="/",
            target="KiroRuntimeService.GenerateAssistantResponse",
            sdk_api="kiroruntime",
            sdk_max_attempts=1,
            extra_headers={
                "x-amzn-codewhisperer-optout": OPTOUT_HEADER_VALUE,
                "x-amzn-kiro-client-attribution": "kiro-ide",
                "x-kiro-attempt": "1;max=1",
            },
        ),
        KiroOperation.LIST_AVAILABLE_MODELS: OperationSpec(
            host_template=IDE_MANAGEMENT_HOST_TEMPLATE,
            path="/",
            target="KiroControlPlaneBearerService.ListAvailableModels",
            sdk_api="kirocontrolplanebearer",
            sdk_max_attempts=3,
        ),
    },
    origin=IDE_ORIGIN,
)

PROFILES: Dict[ClientProfileName, ClientProfile] = {
    ClientProfileName.CLI: CLI_PROFILE,
    ClientProfileName.IDE: IDE_PROFILE,
}


def parse_profile_name(value: Optional[str]) -> ClientProfileName:
    """
    Parse a user-supplied profile name.

    Empty or missing values mean "auto". Matching is case-insensitive and
    ignores surrounding whitespace.

    Args:
        value: Raw value from env var or credentials.json

    Returns:
        Parsed ClientProfileName

    Raises:
        ValueError: If the value is not one of auto/cli/ide
    """
    normalized = (value or "").strip().lower()
    if not normalized:
        return ClientProfileName.AUTO
    try:
        return ClientProfileName(normalized)
    except ValueError:
        allowed = ", ".join(p.value for p in ClientProfileName)
        raise ValueError(
            f"Invalid client profile '{value}'. Allowed values: {allowed}. "
            f"Set KIRO_CLIENT_PROFILE (or 'client_profile' in credentials.json) to one of them."
        ) from None


def detect_profile_for_credentials(source: CredentialSource, is_kiro_ide_file: bool) -> ClientProfileName:
    """
    Pick the profile matching where credentials came from ("auto" mode).

    Rules:
    - kiro-cli SQLite database -> cli
    - JSON written by Kiro IDE (kiro-auth-token.json: Social or Enterprise) -> ide
    - Other JSON (raw AWS SSO cache with clientId/clientSecret, used by kiro-cli) -> cli
    - Plain refresh token from .env (Kiro IDE refresh flow) -> ide

    Args:
        source: Credential source
        is_kiro_ide_file: For JSON_FILE, True if the file was written by Kiro IDE

    Returns:
        CLI or IDE (never AUTO)
    """
    if source == CredentialSource.SQLITE:
        return ClientProfileName.CLI
    if source == CredentialSource.JSON_FILE:
        return ClientProfileName.IDE if is_kiro_ide_file else ClientProfileName.CLI
    return ClientProfileName.IDE


def resolve_client_profile(
    requested: Optional[str],
    source: CredentialSource,
    is_kiro_ide_file: bool,
) -> ClientProfile:
    """
    Resolve the effective client profile for an account.

    Args:
        requested: Raw requested name ("auto", "cli", "ide", empty or None)
        source: Credential source (used in auto mode)
        is_kiro_ide_file: Whether a JSON credentials file was written by Kiro IDE

    Returns:
        Concrete ClientProfile

    Raises:
        ValueError: If requested is not a valid profile name
    """
    name = parse_profile_name(requested)
    if name == ClientProfileName.AUTO:
        resolved = detect_profile_for_credentials(source, is_kiro_ide_file)
        logger.info(f"Client profile: {resolved.value} (auto-detected from {source.value} credentials)")
    else:
        resolved = name
        logger.info(f"Client profile: {resolved.value} (configured)")
    return PROFILES[resolved]
