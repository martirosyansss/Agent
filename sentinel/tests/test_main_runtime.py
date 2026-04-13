"""Тесты runtime helper-функций main.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import _evaluate_binance_permissions


class TestBinancePermissionEvaluation:
    def test_restrictions_allow_spot_only(self):
        ok, reason = _evaluate_binance_permissions(
            {"canTrade": True, "permissions": ["SPOT"]},
            {
                "enableWithdrawals": False,
                "enableFutures": False,
                "enableMargin": False,
            },
        )
        assert ok is True
        assert "verified" in reason

    def test_restrictions_reject_forbidden_rights(self):
        ok, reason = _evaluate_binance_permissions(
            {"canTrade": True, "permissions": ["SPOT"]},
            {
                "enableWithdrawals": True,
                "enableFutures": False,
                "enableMargin": False,
            },
        )
        assert ok is False
        assert "withdraw" in reason

    def test_account_permissions_reject_margin(self):
        ok, reason = _evaluate_binance_permissions(
            {"canTrade": True, "permissions": ["SPOT", "MARGIN"]},
            None,
        )
        assert ok is False
        assert "margin" in reason