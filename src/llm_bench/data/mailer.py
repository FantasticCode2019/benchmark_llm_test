"""SMTP sender with implicit-TLS / STARTTLS auto-detection + retry."""
from __future__ import annotations

import logging
import smtplib
import ssl
import time
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from llm_bench.constants import LOG_NAMESPACE
from llm_bench.domain import AppConfig, EmailConfig
from llm_bench.utils.time_utils import utc_now_naive

log = logging.getLogger(LOG_NAMESPACE)


def _render_subject(template: str, stamp: str) -> str:
    """Substitute ``{date}`` / ``{datetime}`` / ``{stamp}`` placeholders
    in the configured subject. Unknown braces are left untouched (so a
    literal ``{foo}`` in the subject doesn't blow up).
    """
    now = utc_now_naive()
    return (template
            .replace("{date}", now.strftime("%Y-%m-%d"))
            .replace("{datetime}", now.strftime("%Y-%m-%d %H:%M"))
            .replace("{stamp}", stamp))


def _send_once(host: str, port: int, *, timeout: int, use_ssl: bool,
               username: str, password: str, sender: str,
               rcpt: list[str], raw: str) -> None:
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


def _resolve_use_ssl(email: EmailConfig) -> bool:
    """Pick the transport mode: explicit ``use_ssl`` wins, otherwise
    fall back to the port-465-means-implicit-TLS heuristic. 465 hits
    a TLS-only listener with plain SMTP would RST as
    ``SMTPServerDisconnected``, so this default is important.
    """
    if email.use_ssl is not None:
        return email.use_ssl
    return email.smtp_port == 465


def send_email(html: str, json_dump: str, cfg: AppConfig,
               *, stamp: str,
               excel_bytes: bytes | None = None) -> None:
    """Build the MIME message and ship it over SMTP, with retry on
    transport errors only.

    use_ssl heuristic: 465 → implicit TLS (SMTP_SSL); else STARTTLS.
    Hitting 465 with plain SMTP sends EHLO to a TLS-only listener and
    the server RSTs, surfacing as ``SMTPServerDisconnected``.

    Auth/protocol errors are raised on the first try (a retry can't help).

    ``excel_bytes`` is the Ollama-only summary workbook produced by
    :func:`llm_bench.data.excel_report.render_ollama_excel`. When non-
    empty it's attached as a third part alongside the HTML body + JSON
    dump; the official .xlsx MIME type is
    ``application/vnd.openxmlformats-officedocument.spreadsheetml.sheet``.
    Pass None / empty bytes (the default) to suppress the attachment —
    used by runs that contained no Ollama models.
    """
    email = cfg.email
    msg = MIMEMultipart("mixed")
    msg["Subject"] = _render_subject(email.subject, stamp)
    msg["From"] = email.sender
    msg["To"] = email.to

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

    if excel_bytes:
        xlsx = MIMEApplication(
            excel_bytes,
            _subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        xlsx.add_header("Content-Disposition", "attachment",
                        filename=f"llm_bench_ollama_{stamp}.xlsx")
        msg.attach(xlsx)

    use_ssl = _resolve_use_ssl(email)
    retries = max(1, email.smtp_retries)
    rcpt = [a.strip() for a in email.to.split(",") if a.strip()]
    raw = msg.as_string()

    # NOTE on except order: smtplib.SMTPException inherits from OSError, so
    # SMTPAuthenticationError / SMTPException must be caught BEFORE the
    # broad transport tuple — otherwise auth failures would get swallowed
    # by the retry path and burn the whole retry budget.
    last_exc: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            log.info(
                "sending email via %s:%d (ssl=%s, timeout=%ds, "
                "attempt %d/%d) to %s",
                email.smtp_host, email.smtp_port, use_ssl,
                email.smtp_timeout, attempt, retries,
                ", ".join(rcpt))
            _send_once(email.smtp_host, email.smtp_port,
                       timeout=email.smtp_timeout, use_ssl=use_ssl,
                       username=email.username, password=email.password,
                       sender=email.sender, rcpt=rcpt, raw=raw)
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
                time.sleep(min(email.smtp_retry_backoff * attempt, 60))

    assert last_exc is not None
    raise last_exc
