import re

REGEX_PATTERNS = {
    r"User\s+\w+\d+\s+logged\s+(in|out)": "User Action",
    r"Account\s+(?:with\s+)?ID\s+\S+\s+created\s+by": "User Action",
    r"Backup\s+(started|ended|completed\s+successfully)": "System Notification",
    r"System\s+updated\s+to\s+version": "System Notification",
    r"File\s+\S+\s+uploaded\s+successfully\s+by\s+user": "System Notification",
    r"Disk\s+cleanup\s+completed\s+successfully": "System Notification",
    r"System\s+reboot\s+initiated\s+by\s+user": "System Notification",
    r"Scheduled\s+maintenance\s+(started|completed)": "System Notification",
    r"Service\s+\w+\s+restarted\s+successfully": "System Notification",
}

def classify_with_regex(log_message: str) -> str | None:
    """
    Tier 1: Rule-based classifier using regex patterns.
    Returns category label or None if no pattern matches.
    Latency: sub-millisecond.
    """
    for pattern, label in REGEX_PATTERNS.items():
        if re.search(pattern, log_message, re.IGNORECASE):
            return label
    return None


def get_regex_coverage(log_messages: list[str]) -> dict:
    """Measure regex tier coverage on a list of log messages."""
    matched = sum(1 for msg in log_messages if classify_with_regex(msg) is not None)
    return {
        "total": len(log_messages),
        "matched": matched,
        "coverage_pct": round(matched / len(log_messages) * 100, 2),
    }


if __name__ == "__main__":
    test_logs = [
        "User User123 logged in.",
        "Backup completed successfully.",
        "Account with ID 456 created by Admin.",
        "GET /api/v2/resource HTTP/1.1 status: 200",   # should be None
        "Hey bro chill ya!",                            # should be None
    ]
    for log in test_logs:
        result = classify_with_regex(log)
        print(f"[{'✓' if result else '✗'}] {result or 'None':25s} | {log[:60]}")
