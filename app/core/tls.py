"""
TLS utilities for local and container startup.
"""
from __future__ import annotations

import ipaddress
import logging
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.config import get_config

logger = logging.getLogger(__name__)


def generate_self_signed_cert(cert_file: str, key_file: str, common_name: str = "localhost") -> dict:
    cert_path = Path(cert_file)
    key_path = Path(key_file)
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        if shutil.which("openssl"):
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-nodes",
                    "-newkey",
                    "rsa:2048",
                    "-keyout",
                    str(key_path),
                    "-out",
                    str(cert_path),
                    "-days",
                    "3650",
                    "-subj",
                    f"/CN={common_name}",
                    "-addext",
                    f"subjectAltName=DNS:{common_name},DNS:localhost,IP:127.0.0.1,IP:::1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return {
                "generated": True,
                "cert_file": str(cert_path),
                "key_file": str(key_path),
                "common_name": common_name,
                "generator": "openssl",
            }
        raise RuntimeError("Cannot auto-generate TLS certificate: cryptography and openssl are both unavailable")

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "LiveVoiceTranscriptor Dev"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )

    alt_names = {
        "localhost",
        "127.0.0.1",
        "::1",
        common_name,
    }

    san_entries = []
    for item in sorted(alt_names):
        try:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(item)))
        except ValueError:
            san_entries.append(x509.DNSName(item))

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(private_key, hashes.SHA256())
    )

    key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    return {
        "generated": True,
        "cert_file": str(cert_path),
        "key_file": str(key_path),
        "common_name": common_name,
        "generator": "cryptography",
    }


def ensure_tls_assets() -> dict:
    cfg = get_config().server

    if not cfg.tls_enabled:
        return {"enabled": False, "generated": False, "reason": "tls_disabled"}

    cert_exists = Path(cfg.tls_cert_file).is_file()
    key_exists = Path(cfg.tls_key_file).is_file()
    if cert_exists and key_exists:
        return {
            "enabled": True,
            "generated": False,
            "cert_file": cfg.tls_cert_file,
            "key_file": cfg.tls_key_file,
            "reason": "existing_files",
        }

    if not cfg.tls_auto_generate_self_signed:
        missing = []
        if not cert_exists:
            missing.append(cfg.tls_cert_file)
        if not key_exists:
            missing.append(cfg.tls_key_file)
        raise FileNotFoundError(f"TLS is enabled but certificate files are missing: {missing}")

    result = generate_self_signed_cert(
        cert_file=cfg.tls_cert_file,
        key_file=cfg.tls_key_file,
        common_name=cfg.tls_cert_common_name,
    )
    logger.info("Generated self-signed TLS certificate at %s", cfg.tls_cert_file)
    return {"enabled": True, **result, "reason": "auto_generated_self_signed"}


def main() -> None:
    info = ensure_tls_assets()
    print(info)


if __name__ == "__main__":
    main()
