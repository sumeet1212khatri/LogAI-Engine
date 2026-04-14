"""
processor_regex.py — Tier 1: Rule-based Classifier

Target coverage: 40%+ (up from 15%)
Latency: sub-millisecond per log

New pattern groups added:
  - HTTP request/response logs   (was completely missing!)
  - Auth / credential events     (login failures, MFA, lockouts)
  - System/infra events          (disk, CPU, memory, cron)
  - Network / firewall events    (IP block, port scan)
  - Structured error codes       (ERROR, CRITICAL prefix logs)
"""
from __future__ import annotations
import re
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Pattern registry: (compiled_pattern, label)
# Order matters — more specific patterns FIRST to avoid mis-labeling.
# ---------------------------------------------------------------------------
_RAW_PATTERNS: list[tuple[str, str]] = [

    # ── HTTP Status ─────────────────────────────────────────────────────────
    # Covers: GET/POST/PUT/DELETE/PATCH + status code in request line
    (r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+\S+\s+HTTP/\d", "HTTP Status"),
    # Nova / OpenStack style
    (r"nova\.\S+\s+(GET|POST|PUT|DELETE)\s+\S+\s+HTTP/\d", "HTTP Status"),
    # Status code only style: "returned HTTP 200" or "status: 404"
    (r"\bstatus[:\s]+\d{3}\b", "HTTP Status"),
    (r"\breturned\s+HTTP\s+\d{3}\b", "HTTP Status"),
    (r"\bHTTP\s+status\s+code\s*[:-]?\s*\d{3}\b", "HTTP Status"),
    # API response style
    (r"\bAPI\s+(call|request)\s+\S+\s+completed\s+with\s+status\s+\d{3}", "HTTP Status"),
    (r"\bEndpoint\s+\S+\s+responded\s+with\s+code\s+\d{3}", "HTTP Status"),

    # ── Security Alert ──────────────────────────────────────────────────────
    # Brute force / login failures
    (r"(multiple\s+)?(bad\s+|failed?\s+)?login\s+(failure|attempt|failures)", "Security Alert"),
    (r"brute[\s_-]force\s+(login|attack|attempt)", "Security Alert"),
    # Unauthorized access
    (r"unauthorized\s+(access|admin|privilege|attempt)", "Security Alert"),
    (r"access\s+denied\s+(for|to)\s+(user|ip|host)", "Security Alert"),
    # Privilege escalation
    (r"(admin\s+)?access\s+escalation\s+detected", "Security Alert"),
    (r"privilege\s+(elev|escalat)", "Security Alert"),
    # IP blocking / suspicious traffic
    (r"IP\s+\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\s+blocked", "Security Alert"),
    (r"(suspicious|anomalous)\s+(login|traffic|activity|request)", "Security Alert"),
    (r"potential\s+(DDoS|attack|breach|intrusion)", "Security Alert"),
    (r"security\s+breach\s+suspected", "Security Alert"),
    (r"(API\s+security\s+breach|bypass\s+API\s+security)", "Security Alert"),
    (r"port\s+scan\s+(detected|attempt)", "Security Alert"),

    # ── User Action ─────────────────────────────────────────────────────────
    (r"User\s+\w+\d*\s+logged\s+(in|out)", "User Action"),
    (r"Account\s+(with\s+)?ID\s+\S+\s+created\s+by", "User Action"),
    (r"User\s+\w+\d*\s+(updated\s+profile|changed\s+password|enabled\s+two|downloaded|exported)", "User Action"),
    (r"(New\s+user|user\s+\w+\d*)\s+registered", "User Action"),
    (r"Account\s+\S+\s+deleted\s+by\s+(administrator|admin)", "User Action"),
    (r"User\s+\w+\d*\s+(tried|attempted)", "User Action"),

    # ── System Notification ─────────────────────────────────────────────────
    # Backup events
    (r"Backup\s+(started|ended|completed\s+successfully|failed|aborted)", "System Notification"),
    (r"System\s+updated\s+to\s+version", "System Notification"),
    (r"File\s+\S+\s+uploaded\s+successfully\s+by\s+user", "System Notification"),
    (r"Disk\s+cleanup\s+completed\s+successfully", "System Notification"),
    (r"System\s+reboot\s+initiated\s+by\s+user", "System Notification"),
    (r"Scheduled\s+maintenance\s+(started|completed)", "System Notification"),
    (r"Service\s+\w+\s+restarted\s+successfully", "System Notification"),
    # NEW: cache, cron, health check, cert, log rotation
    (r"Cache\s+cleared\s+successfully", "System Notification"),
    (r"Log\s+rotation\s+completed", "System Notification"),
    (r"Health\s+check\s+(passed|failed)\s+for\s+service", "System Notification"),
    (r"Certificate\s+(renewed|expired|revoked)\s+successfully", "System Notification"),
    (r"Cron\s+job\s+\S+\s+(executed|failed|completed)\s+successfully", "System Notification"),
    (r"(Disk|Storage)\s+(usage|space)\s+(at|reached|exceeded)\s+\d+%", "System Notification"),
    (r"CPU\s+usage\s+at\s+\d+%", "System Notification"),
    (r"Memory\s+(usage|limit)\s+(at|reached|exceeded)\s+\d+%", "System Notification"),
    # Deployment / config
    (r"Deployment\s+(of|for)\s+\S+\s+(completed|failed|started)", "System Notification"),
    (r"Configuration\s+(reloaded|updated|applied)\s+successfully", "System Notification"),

    # ── Error ───────────────────────────────────────────────────────────────
    (r"\bERROR\b.*\b(exception|failed|failure|crash|timeout|unavailable)\b", "Error"),
    (r"System\s+crashed\s+due\s+to", "Error"),
    (r"(connection|request|task|job)\s+(timed?\s*out|timeout)", "Error"),
    (r"service\s+\S+\s+(is\s+down|unavailable|unreachable)", "Error"),
    (r"database\s+connection\s+(failed|refused|lost|dropped)", "Error"),
    (r"disk\s+(I/O\s+)?failure", "Error"),
    (r"driver\s+error(s)?\s+(when|during|on)", "Error"),
    (r"(replication|sync)\s+task\s+(did\s+not\s+complete|failed)", "Error"),
    (r"null\s+pointer|segmentation\s+fault|stack\s+overflow", "Error"),

    # ── Critical Error ──────────────────────────────────────────────────────
    (r"\bCRITICAL\b", "Critical Error"),
    (r"(FATAL|PANIC)\b", "Critical Error"),
    (r"(data\s+loss|data\s+corruption)\s+(detected|occurred)", "Critical Error"),
    (r"(cluster|node|shard)\s+(failure|crashed|went\s+down)", "Critical Error"),
    (r"(catastrophic|unrecoverable)\s+(failure|error)", "Critical Error"),
    (r"kernel\s+panic", "Critical Error"),
    (r"out[\s-]of[\s-](memory|disk)\s+(error|killed|OOM)", "Critical Error"),
]

# Pre-compile all patterns at import time (not per-call)
REGEX_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), label)
    for pat, label in _RAW_PATTERNS
]


def classify_with_regex(log_message: str) -> Optional[str]:
    """
    Tier 1: Rule-based classifier.
    Returns category label, or None if no pattern matches.
    Latency: sub-millisecond (patterns pre-compiled at import).
    """
    for pattern, label in REGEX_PATTERNS:
        if pattern.search(log_message):
            return label
    return None


def get_regex_coverage(log_messages: list[str]) -> dict:
    """Measure regex tier coverage and per-label breakdown."""
    label_counts: dict[str, int] = {}
    missed = 0

    for msg in log_messages:
        label = classify_with_regex(msg)
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1
        else:
            missed += 1

    total   = len(log_messages)
    matched = total - missed

    return {
        "total":        total,
        "matched":      matched,
        "missed":       missed,
        "coverage_pct": round(matched / total * 100, 2) if total else 0.0,
        "label_breakdown": label_counts,
    }


def benchmark_regex(log_messages: list[str], runs: int = 3) -> dict:
    """Measure regex tier latency (p50 / p95 / p99) over multiple runs."""
    import statistics
    per_log_ms: list[float] = []

    for _ in range(runs):
        for msg in log_messages:
            t0 = time.perf_counter()
            classify_with_regex(msg)
            per_log_ms.append((time.perf_counter() - t0) * 1000)

    per_log_ms.sort()
    return {
        "p50_ms":  round(statistics.median(per_log_ms), 4),
        "p95_ms":  round(per_log_ms[int(len(per_log_ms) * 0.95)], 4),
        "p99_ms":  round(per_log_ms[int(len(per_log_ms) * 0.99)], 4),
        "mean_ms": round(statistics.mean(per_log_ms), 4),
    }


# ── CLI self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases: list[tuple[str, str]] = [
        # HTTP
        ("GET /api/v2/resource HTTP/1.1 status: 200 len: 1583 time: 0.19", "HTTP Status"),
        ("POST /v1/users HTTP/1.1 status: 201 len: 42 time: 0.05", "HTTP Status"),
        ("nova.osapi_compute.wsgi.server GET /v2/servers/detail HTTP/1.1 status: 404", "HTTP Status"),
        # Security
        ("Multiple login failures occurred on user 6454 account", "Security Alert"),
        ("IP 192.168.133.114 blocked due to potential attack", "Security Alert"),
        ("Brute force login attempt from 10.0.0.5 detected", "Security Alert"),
        ("Admin access escalation detected for user 9429", "Security Alert"),
        # User Action
        ("User User12345 logged in.", "User Action"),
        ("Account with ID 456 created by Admin.", "User Action"),
        # System Notification
        ("Backup completed successfully.", "System Notification"),
        ("CPU usage at 98% for the last 10 minutes on node-7", "System Notification"),
        ("Health check passed for service payments-api", "System Notification"),
        # Error
        ("System crashed due to disk I/O failure on node-3", "Error"),
        ("Database connection failed after 3 retries", "Error"),
        # Critical
        ("CRITICAL: data corruption detected on shard-14", "Critical Error"),
        ("kernel panic: not syncing: VFS: unable to mount root fs", "Critical Error"),
        # Should be None (unmatched)
        ("The 'BulkEmailSender' feature will be deprecated in v5.0.", None),
        ("Case escalation for ticket 7324 failed.", None),
    ]

    correct = 0
    print(f"{'Expected':<22} {'Got':<22} {'✓/✗'} | Log")
    print("─" * 100)
    for log, expected in test_cases:
        got = classify_with_regex(log)
        ok  = got == expected
        correct += ok
        icon = "✓" if ok else "✗"
        print(f"{str(expected):<22} {str(got):<22} {icon}   | {log[:55]}")

    print(f"\n{correct}/{len(test_cases)} correct")

    # Coverage demo
    all_logs = [log for log, _ in test_cases]
    cov = get_regex_coverage(all_logs)
    print(f"\nCoverage: {cov['coverage_pct']}%  ({cov['matched']}/{cov['total']} matched)")
    print("Label breakdown:", cov["label_breakdown"])

    # Latency benchmark
    lat = benchmark_regex(all_logs * 100)
    print(f"\nLatency (p50/p95/p99): {lat['p50_ms']}ms / {lat['p95_ms']}ms / {lat['p99_ms']}ms")
