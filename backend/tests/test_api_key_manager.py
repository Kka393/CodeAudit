"""
Tests 8, 9, 10: APIKeyManager round-robin rotation and failover.
Uses mock keys -- never prints real key values (there are none here).
"""

import time

import pytest

from app.llm.api_key_manager import APIKeyManager, NoAvailableKeyError

MOCK_KEYS = ["KEY1", "KEY2", "KEY3", "KEY4", "KEY5"]


def test_round_robin_rotation_wraps_around():
    manager = APIKeyManager(keys=MOCK_KEYS, cooldown_seconds=60)

    seen = [manager.get_next_key() for _ in range(7)]
    assert seen == ["KEY1", "KEY2", "KEY3", "KEY4", "KEY5", "KEY1", "KEY2"]


def test_failed_key_goes_into_cooldown_and_is_skipped():
    manager = APIKeyManager(keys=MOCK_KEYS, cooldown_seconds=60)

    first = manager.get_next_key()  # KEY1
    assert first == "KEY1"

    second = manager.get_next_key()  # KEY2
    assert second == "KEY2"

    manager.report_failure("KEY3")  # simulate KEY3 rate-limited before its turn

    # position is now at KEY3; it should be skipped since it's cooling down
    third = manager.get_next_key()
    assert third == "KEY4"


def test_all_keys_in_cooldown_raises():
    manager = APIKeyManager(keys=["ONLY_KEY"], cooldown_seconds=60)
    manager.get_next_key()
    manager.report_failure("ONLY_KEY")

    with pytest.raises(NoAvailableKeyError):
        manager.get_next_key()


def test_cooldown_expires_after_wait():
    manager = APIKeyManager(keys=["ONLY_KEY"], cooldown_seconds=0)  # instant expiry for the test
    manager.get_next_key()
    manager.report_failure("ONLY_KEY")
    time.sleep(0.01)
    assert manager.get_next_key() == "ONLY_KEY"


def test_empty_key_list_raises_on_use():
    manager = APIKeyManager(keys=[])
    with pytest.raises(NoAvailableKeyError):
        manager.get_next_key()
