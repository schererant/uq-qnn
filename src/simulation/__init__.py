from simulation.logger import SimulationLogger
from simulation.runner import SimulationRunner

def compute_n_swipe(
    t_phase_ms: float,
    f_laser_khz: float,
    det_window_us: float,
    max_swipe: int = 201,
) -> int:
    """
    Translates hardware-timing limits into a safe, odd swipe count.
    
    Args:
        t_phase_ms (float): Heater settle time in milliseconds
        f_laser_khz (float): Laser repetition rate in kHz
        det_window_us (float): Detector integration window in microseconds
        max_swipe (int): Maximum allowed swipe count
        
    Returns:
        int: Odd integer swipe count, capped at max_swipe
    """
    if t_phase_ms <= 0 or f_laser_khz <= 0 or det_window_us <= 0:
        raise ValueError("Timing inputs must be positive.")

    period_laser_us = 1_000 / f_laser_khz  # µs
    slot_us = max(period_laser_us, det_window_us)
    slots_total = int((t_phase_ms * 1_000) // slot_us)  # integer slots
    n_swipe = max(1, 2 * (slots_total // 2) + 1)  # force odd
    return min(n_swipe, max_swipe)

__all__ = ['SimulationLogger', 'SimulationRunner', 'compute_n_swipe'] 