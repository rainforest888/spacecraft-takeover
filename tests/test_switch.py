# tests/test_switch.py
import numpy as np
from algorithms.switch_manager import SwitchManager


def test_switch_manager_starts_in_weakening():
    sm = SwitchManager()
    assert sm.phase == "weakening"


def test_switch_when_torque_weakened():
    sm = SwitchManager(weak_threshold=0.5, sustain_steps=5)
    sm.peak_torque = 10.0
    for _ in range(6):
        result = sm.update(tau_target_mag=0.3, attitude_error=0.1, self_fuel=0.5)
    assert sm.phase in ("lqr", "transition")


def test_switch_on_desperation():
    sm = SwitchManager(weak_threshold=0.5, sustain_steps=100)
    sm.peak_torque = 10.0
    result = sm.update(tau_target_mag=9.0, attitude_error=0.1, self_fuel=0.05)
    assert result or sm.phase != "weakening"


def test_blend_factor_decays():
    sm = SwitchManager(blend_steps=10)
    sm.phase = "transition"
    sm._transition_step = 0
    alpha = None
    for _ in range(11):
        alpha = sm.get_blend_alpha()
    assert alpha is not None and alpha <= 0.05


def test_full_weakening_to_lqr_flow():
    sm = SwitchManager(weak_threshold=0.5, sustain_steps=3, blend_steps=5)
    sm.peak_torque = 10.0
    assert sm.phase == "weakening"
    for i in range(4):
        sm.update(tau_target_mag=0.2, attitude_error=0.05, self_fuel=0.8)
    assert sm.phase in ("lqr", "transition")
