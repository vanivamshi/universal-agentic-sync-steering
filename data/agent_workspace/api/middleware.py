from config.settings import RATE_LIMIT


def check_rate(count: int) -> bool:
    return count < RATE_LIMIT
