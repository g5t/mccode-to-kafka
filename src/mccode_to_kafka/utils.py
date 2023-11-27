from datetime import datetime


def datetime_to_ns_since_epoch(dt: datetime) -> int:
    from datetime import timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def ns_since_epoch_to_datetime(ns: int) -> datetime:
    from datetime import timezone
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)


def now_in_ns_since_epoch() -> int:
    from datetime import timezone
    return datetime_to_ns_since_epoch(datetime.now(timezone.utc))
