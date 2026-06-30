# algorithms/switch_manager.py
"""State machine for TD3->LQR phase transition based on observable torque decay."""


class SwitchManager:
    PHASE_WEAKENING = "weakening"
    PHASE_TRANSITION = "transition"
    PHASE_LQR = "lqr"

    def __init__(self, weak_threshold=0.10, sustain_steps=30,
                 blend_steps=50, desperation_threshold=0.10):
        self.weak_threshold = weak_threshold
        self.sustain_steps = sustain_steps
        self.blend_steps = blend_steps
        self.desperation_threshold = desperation_threshold

        self.phase = self.PHASE_WEAKENING
        self.peak_torque = 0.0
        self._low_torque_counter = 0
        self._transition_step = 0

    def update(self, tau_target_mag, attitude_error, self_fuel):
        """Returns True when switch to LQR is triggered."""
        self.peak_torque = max(self.peak_torque, tau_target_mag)
        weak_thresh_abs = self.weak_threshold * max(self.peak_torque, 1e-6)

        if self.phase == self.PHASE_WEAKENING:
            triggered = False

            # Criterion 1: sustained low target torque
            if tau_target_mag < weak_thresh_abs:
                self._low_torque_counter += 1
            else:
                self._low_torque_counter = 0

            if self._low_torque_counter >= self.sustain_steps:
                triggered = True

            # Criterion 2: desperation — self fuel almost gone
            if self_fuel < self.desperation_threshold:
                triggered = True

            if triggered:
                self.phase = self.PHASE_TRANSITION
                self._transition_step = 0
                return True

        return False

    def get_blend_alpha(self):
        """Transition blend factor: 1.0 = pure TD3, 0.0 = pure LQR."""
        if self.phase == self.PHASE_WEAKENING:
            return 1.0
        if self.phase == self.PHASE_LQR:
            return 0.0
        alpha = 1.0 - self._transition_step / max(self.blend_steps, 1)
        self._transition_step += 1
        if self._transition_step >= self.blend_steps:
            self.phase = self.PHASE_LQR
        return max(0.0, alpha)
