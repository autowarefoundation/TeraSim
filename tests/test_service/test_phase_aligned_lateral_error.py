import unittest
from types import SimpleNamespace

from terasim_service.utils.carla import cosim as carla_cosim
from terasim_service.utils.carla.cosim import CarlaCosim


class PhaseAlignedLateralErrorTest(unittest.TestCase):
    def test_uses_target_time_and_front_center(self):
        current = carla_cosim.carla.Transform(
            carla_cosim.carla.Location(x=0.0, y=0.0, z=0.0),
            carla_cosim.carla.Rotation(yaw=0.0),
        )
        desired = carla_cosim.carla.Transform(
            carla_cosim.carla.Location(x=1.0, y=1.0, z=0.0),
            carla_cosim.carla.Rotation(yaw=0.0),
        )
        error = CarlaCosim._phase_aligned_front_lateral_error(
            current,
            desired,
            1.0,
            SimpleNamespace(x=10.0, y=0.0),
            0.0,
            0.1,
        )
        self.assertAlmostEqual(error, 1.0)


if __name__ == "__main__":
    unittest.main()
