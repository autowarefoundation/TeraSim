from terasim.overlay import traci
from terasim.vehicle.factories.vehicle_factory import VehicleFactory
from terasim.vehicle.vehicle import Vehicle

from .av_sumo_lane_change_decision_model import AVSumoLaneChangeDecisionModel
from .conflict_generation_model import ConflictGenerationModel
from .emergency_vehicle_model import EmergencyVehicleModel
from .nde_controller import NDEController
from .nde_decision_model import NDEDecisionModel
from .nde_ego_sensor import NDEEgoSensor


class NDEVehicleFactory(VehicleFactory):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg

    def create_vehicle(self, veh_id, simulator):
        """Create a vehicle object.

        Args:
            veh_id (str): Vehicle ID.
            simulator (Simulator): Simulator object.

        Returns:
            Vehicle: Vehicle object.
        """
        sensor_list = [NDEEgoSensor(cache=True, cache_history=True, cache_history_duration=1)]
        av_control_mode = (
            str(self.cfg.get("av_control_mode", "external")).strip().lower()
            if self.cfg is not None
            else "external"
        )
        sumo_controlled_av = veh_id == "AV" and av_control_mode == "sumo"
        if sumo_controlled_av:
            # SUMO computes the ordinary lane-change need, while the adapter
            # validates and emits one explicit changeLaneRelative request.
            # Keeping laneChangeMode=0 prevents a second autonomous maneuver.
            decision_model = AVSumoLaneChangeDecisionModel(
                cfg=self.cfg,
                MOBIL_lc_flag=True,
                stochastic_acc_flag=False,
                reroute=False,
                dynamically_change_vtype=False,
            )
        elif veh_id == "AV":
            # External mode preserves the historical AV behavior and keeps it
            # outside NADE adversarial sampling.
            decision_model = NDEDecisionModel(
                MOBIL_lc_flag=True,
                stochastic_acc_flag=False,
                reroute=False,
                dynamically_change_vtype=False,
            )
        elif traci.vehicle.getVehicleClass(veh_id) == "emergency":
            decision_model = EmergencyVehicleModel()
        else:
            decision_model = ConflictGenerationModel(
                MOBIL_lc_flag=True,
                stochastic_acc_flag=False,
                dynamically_change_vtype=False,
                cfg=self.cfg,
            )

        controller = NDEController(
            simulator,
            preserve_sumo_speed_control=sumo_controlled_av,
            release_lane_change_mode=0 if veh_id == "AV" else 1621,
        )
        return Vehicle(
            veh_id,
            simulator,
            sensors=sensor_list,
            decision_model=decision_model,
            controller=controller,
        )
