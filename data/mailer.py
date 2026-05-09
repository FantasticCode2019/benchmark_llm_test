"""SMTP sender with implicit-TLS / STARTTLS auto-detection + retry."""
from __future__ import annotations

import logging
import smtplib
import ssl
import time
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

log = logging.getLogger("llm_bench")


def _send_once(host: str, port: int, *, timeout: int, use_ssl: bool,
               username: str, password: str, sender: str,
               rcpt: list, raw: str) -> None:
    """One SMTP attempt — implicit TLS (465) vs STARTTLS (587)."""
    ctx = ssl.create_default_context()
    if use_ssl:
        with smtplib.SMTP_SSL(host, port, timeout=timeout, context=ctx) as s:
            s.ehlo()
            s.login(username, password)
            s.sendmail(sender, rcpt, raw)
    else:
        with smtplib.SMTP(host, port, timeout=timeout) as s:
            s.ehlo()
            s.starttls(context=ctx)
            s.ehlo()  # re-EHLO after STARTTLS (RFC 3207)
            s.login(username, password)
            s.sendmail(sender, rcpt, raw)


def send_email(html: str, json_dump: str, cfg: dict, *, stamp: str) -> None:
    """Build the MIME message and ship it over SMTP, with retry on
    transport errors only.

    use_ssl heuristic: 465 → implicit TLS (SMTP_SSL); else STARTTLS.
    Hitting 465 with plain SMTP sends EHLO to a TLS-only listener and
    the server RSTs, surfacing as `SMTPServerDisconnected`.

    Auth/protocol errors are raised on the first try (a retry can't help).
    """
    smtp_cfg = cfg["email"]
    msg = MIMEMultipart("mixed")
    msg["Subject"] = smtp_cfg.get(
        "subject", f"Olares LLM benchmark {datetime.utcnow().date()}")
    msg["From"] = smtp_cfg["from"]
    msg["To"] = smtp_cfg["to"]

    body = MIMEMultipart("alternative")
    body.attach(MIMEText(
        "Olares LLM benchmark results — see HTML body or attached JSON.",
        "plain", _charset="utf-8"))
    body.attach(MIMEText(html, "html", _charset="utf-8"))
    msg.attach(body)

    att = MIMEApplication(json_dump.encode("utf-8"), _subtype="json")
    att.add_header("Content-Disposition", "attachment",
                   filename=f"llm_bench_{stamp}.json")
    msg.attach(att)

    host = smtp_cfg["smtp_host"]
    port = int(smtp_cfg["smtp_port"])
    timeout = int(smtp_cfg.get("smtp_timeout", 120))
    retries = max(1, int(smtp_cfg.get("smtp_retries", 3)))
    backoff = int(smtp_cfg.get("smtp_retry_backoff", 5))
    use_ssl_cfg = smtp_cfg.get("use_ssl")
    use_ssl = bool(use_ssl_cfg) if use_ssl_cfg is not None else (port == 465)

    rcpt = [a.strip() for a in str(smtp_cfg["to"]).split(",") if a.strip()]
    raw = msg.as_string()

    # NOTE on except order: smtplib.SMTPException inherits from OSError, so
    # SMTPAuthenticationError / SMTPException must be caught BEFORE the
    # broad transport tuple — otherwise auth failures would get swallowed
    # by the retry path and burn the whole retry budget.
    last_exc: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            log.info(
                "sending email via %s:%d (ssl=%s, timeout=%ds, "
                "attempt %d/%d) to %s",
                host, port, use_ssl, timeout, attempt, retries,
                ", ".join(rcpt))
            _send_once(host, port, timeout=timeout, use_ssl=use_ssl,
                       username=smtp_cfg["username"],
                       password=smtp_cfg["password"],
                       sender=smtp_cfg["from"], rcpt=rcpt, raw=raw)
            log.info("email sent")
            return
        except smtplib.SMTPAuthenticationError:
            raise  # bad credentials — retrying won't help
        except smtplib.SMTPException:
            raise  # protocol-level 5xx etc. — won't be rescued by a retry
        except (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError,
                TimeoutError, ConnectionError, OSError) as exc:
            last_exc = exc
            log.warning("smtp attempt %d/%d failed: %s: %s",
                        attempt, retries, type(exc).__name__, exc)
            if attempt < retries:
                time.sleep(min(backoff * attempt, 60))

    assert last_exc is not None
    raise last_exc
