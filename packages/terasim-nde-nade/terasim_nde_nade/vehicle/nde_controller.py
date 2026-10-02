import numpy as np
import terasim.utils as utils
from addict import Dict
from loguru import logger
from terasim.agent.agent_controller import AgentController
from terasim.overlay import traci

from ..utils import (
    CommandType,
    NDECommand,
    interpolate_future_trajectory,
)


def get_all_routes():
    """Get all routes in the network.

    Returns:
        list: List of all routes in the network.
    """
    return traci.route.getIDList()


def get_all_route_edges():
    """Get all route edges in the network.

    Returns:
        dict: Dictionary of all route edges in the network.
    """
    all_routes = get_all_routes()
    all_route_edges = {}
    for route in all_routes:
        all_route_edges[route] = traci.route.getEdges(route)
    return all_route_edges


class NDEController(AgentController):
    def __init__(
        self,
        simulator,
        params=None,
        *,
        preserve_sumo_speed_control=False,
        release_lane_change_mode=1621,
    ):
        self.is_busy = False
        self.cached_control_command = None  # this is a dict, containing the control command for the vehicle with the timestep information
        self.preserve_sumo_speed_control = bool(preserve_sumo_speed_control)
        self.release_lane_change_mode = int(release_lane_change_mode)
        self.lane_change_request_count = 0
        self.last_lane_change_request = None
        self.last_command_end_reason = ""
        self._last_request_generation = 0
        return super().__init__(simulator, control_command_schema=NDECommand, params=params)

    def _update_controller_status(self, veh_id, current_time=None):
        """Refresh the state of the controller. This function will be called at each timestep as far as vehicle is still in the simulator, even if the vehicle is not controlled.

        Args:
            veh_id (str): Vehicle ID.
            current_time (float, optional): Current simulation time. Defaults to None.
        """
        # if the controller is busy, detect if the current simulation time - the time of the cached control command is greater than the duration of the control command, then the controller is not busy anymore
        if self.is_busy:
            current_time = traci.simulation.getTime() if current_time is None else current_time
            if (
                current_time - self.cached_control_command.timestep
                > self.cached_control_command.cached_command.duration
            ):
                self.is_busy = False
                self.cached_control_command = None
                self.last_command_end_reason = "duration_elapsed"
                self._restore_control_modes(veh_id)

    def execute_control_command(self, veh_id, control_command, obs_dict):
        """Vehicle acts based on the input action.

        Args:
            veh_id (str): Vehicle ID.
            control_command (NDECommand): Control command.
            obs_dict (dict): Observation of the ego agent.
        """
        generation = int((control_command.info or {}).get("request_generation", 0))
        if generation and generation <= self._last_request_generation and not self.is_busy:
            return
        if not self.is_busy:
            if control_command.command_type == CommandType.DEFAULT:
                # all_checks_on(veh_id)
                return
            elif control_command.command_type == CommandType.CUSTOM:
                self.cached_control_command = Dict(
                    {
                        "timestep": traci.simulation.getTime(),
                        "cached_command": control_command,
                    }
                )
                self.execute_control_command_onestep(
                    veh_id, self.cached_control_command, obs_dict, first_step=True
                )
                return
            else:
                self._prepare_control_modes(veh_id)
                # other commands will have duration, which will keep the controller busy
                self.is_busy = True
                # if the control command is a trajectory, then interpolate the trajectory
                control_command = interpolate_control_command(control_command, obs_dict)
                self.cached_control_command = Dict(
                    {
                        "timestep": traci.simulation.getTime(),
                        "cached_command": control_command,
                    }
                )
                self.execute_control_command_onestep(
                    veh_id, self.cached_control_command, obs_dict, first_step=True
                )
        else:
            self.execute_control_command_onestep(
                veh_id, self.cached_control_command, obs_dict, first_step=False
            )

    def execute_control_command_onestep(
        self, veh_id, cached_control_command, obs_dict, first_step=False
    ):
        """Execute the control command for one step.

        Args:
            veh_id (str): Vehicle ID.
            cached_control_command (dict): Cached control command.
            obs_dict (dict): Observation of the ego agent.
            first_step (bool, optional): Flag to indicate if this is the first step. Defaults to False.
        """
        if cached_control_command["cached_command"].command_type == CommandType.CUSTOM:
            if (
                cached_control_command["cached_command"].custom_control_command is not None
                and cached_control_command["cached_command"].custom_execute_control_command
                is not None
            ):
                cached_control_command["cached_command"].custom_execute_control_command(
                    veh_id,
                    cached_control_command["cached_command"].custom_control_command,
                    obs_dict,
                )
                return
            else:
                logger.error(
                    "Custom control command or Custom control command execution is not defined"
                )
                return

        if cached_control_command["cached_command"].command_type == CommandType.TRAJECTORY:
            # pass
            self.execute_trajectory_command(
                veh_id, cached_control_command["cached_command"], obs_dict
            )
        elif (
            cached_control_command["cached_command"].command_type == CommandType.LEFT
            or cached_control_command["cached_command"].command_type == CommandType.RIGHT
        ):
            self.execute_lane_change_command(
                veh_id,
                cached_control_command["cached_command"],
                obs_dict,
                first_step=first_step,
            )
        elif cached_control_command["cached_command"].command_type == CommandType.ACCELERATION:
            self.execute_acceleration_command(
                veh_id, cached_control_command["cached_command"], obs_dict
            )
        else:
            logger.error("Invalid command type")
        return

    @staticmethod
    def execute_trajectory_command(veh_id, control_command, obs_dict):
        """Execute the trajectory command.

        Args:
            veh_id (str): Vehicle ID.
            control_command (NDECommand): Control command.
            obs_dict (dict): Observation of the ego agent.
        """
        assert control_command.command_type == CommandType.TRAJECTORY
        # get the closest timestep trajectory point in control_command.trajectory to current timestep
        trajectory_array = control_command.future_trajectory
        next_timestep = traci.simulation.getTime() + traci.simulation.getDeltaT()
        closest_timestep_trajectory = min(
            trajectory_array, key=lambda x: abs(x[-1] - next_timestep)
        )
        # set the position of the vehicle to the closest timestep trajectory point
        traci.vehicle.moveToXY(
            vehID=veh_id,
            edgeID="",
            laneIndex=-1,
            x=closest_timestep_trajectory[0],
            y=closest_timestep_trajectory[1],
            angle=closest_timestep_trajectory[2],
            keepRoute=control_command.keep_route_mode,
        )
        logger.info(
            f"Setting position of {veh_id} to {closest_timestep_trajectory[0], closest_timestep_trajectory[1]}, current position is {traci.vehicle.getPosition(veh_id)}"
        )
        traci.vehicle.setPreviousSpeed(veh_id, closest_timestep_trajectory[3])

    def execute_lane_change_command(self, veh_id, control_command, obs_dict, first_step=False):
        """Execute the lane change command.

        Args:
            veh_id (str): Vehicle ID.
            control_command (NDECommand): Control command.
            obs_dict (dict): Observation of the ego agent.
            first_step (bool, optional): Flag to indicate if this is the first step. Defaults to False.
        """
        assert (
            control_command.command_type == CommandType.LEFT
            or control_command.command_type == CommandType.RIGHT
        )
        if first_step:  # only execute lane change command once
            indexOffset = 1 if control_command.command_type == CommandType.LEFT else -1
            traci.vehicle.changeLaneRelative(veh_id, indexOffset, control_command.duration)
            self._last_request_generation = int(
                (control_command.info or {}).get("request_generation", 0)
            )
            self.lane_change_request_count += 1
            try:
                lane_change_state = traci.vehicle.getLaneChangeState(veh_id, indexOffset)
                model_state = int(lane_change_state[0])
                post_state = int(lane_change_state[1])
            except Exception:
                model_state = None
                post_state = None
            try:
                effective_lane_change_mode = int(traci.vehicle.getLaneChangeMode(veh_id))
            except Exception:
                effective_lane_change_mode = None
            command_info = dict(control_command.info or {})
            self.last_lane_change_request = {
                "count": self.lane_change_request_count,
                "simulation_time": float(traci.simulation.getTime()),
                "direction": "left" if indexOffset > 0 else "right",
                "index_offset": indexOffset,
                "duration": float(control_command.duration),
                "decision_source": command_info.get("decision_source", "nde"),
                "source_lane_id": command_info.get("source_lane_id", ""),
                "target_lane_id": command_info.get("target_lane_id", ""),
                "model_lane_change_state": command_info.get("model_lane_change_state", model_state),
                "post_lane_change_state": command_info.get("post_lane_change_state", post_state),
                "effective_lane_change_mode": effective_lane_change_mode,
                "reason_bits": tuple(command_info.get("reason_bits", ())),
                "blocked": bool(command_info.get("blocked", False)),
                "could_change_lane": command_info.get("could_change_lane"),
                "lane_remaining": command_info.get("lane_remaining"),
                "required_distance": command_info.get("required_distance"),
                "request_generation": command_info.get("request_generation"),
                "route_required": bool(command_info.get("route_required", False)),
                "evaluation_reason": command_info.get("evaluation_reason", ""),
            }
            self.last_command_end_reason = ""
            logger.info(
                "NDE lane-change request: actor={} direction={} duration={} "
                "source={} target={} count={}",
                veh_id,
                self.last_lane_change_request["direction"],
                control_command.duration,
                self.last_lane_change_request["source_lane_id"],
                self.last_lane_change_request["target_lane_id"],
                self.lane_change_request_count,
            )

    @staticmethod
    def execute_acceleration_command(veh_id, control_command, obs_dict):
        """Execute the acceleration command.

        Args:
            veh_id (str): Vehicle ID.
            control_command (NDECommand): Control command.
            obs_dict (dict): Observation of the ego agent.
        """
        # logger.critical("the acceleration command should not be executed")
        assert control_command.command_type == CommandType.ACCELERATION
        acceleration = control_command.acceleration
        final_speed = obs_dict["ego"]["velocity"] + acceleration * utils.get_step_size()
        final_speed = 0 if final_speed < 0 else final_speed
        traci.vehicle.setSpeed(veh_id, final_speed)

    def _prepare_control_modes(self, veh_id):
        """Disable autonomous lateral changes without stealing AV car-following."""
        speed_mode = 31 if self.preserve_sumo_speed_control else 32
        traci.vehicle.setSpeedMode(veh_id, speed_mode)
        traci.vehicle.setLaneChangeMode(veh_id, 0)

    def _restore_control_modes(self, veh_id):
        traci.vehicle.setSpeedMode(veh_id, 31)
        traci.vehicle.setLaneChangeMode(veh_id, self.release_lane_change_mode)

    def reset_control_state(self, reason="reset"):
        self.is_busy = False
        self.cached_control_command = None
        self.last_lane_change_request = None
        self.lane_change_request_count = 0
        self.last_command_end_reason = reason
        self._last_request_generation = 0

    @staticmethod
    def all_checks_on(veh_id):
        """Turn on all checks for the vehicle.

        Args:
            veh_id (str): Vehicle ID.
        """
        traci.vehicle.setSpeedMode(veh_id, 31)
        traci.vehicle.setLaneChangeMode(veh_id, 1621)

    @staticmethod
    def all_checks_off(veh_id):
        """Turn off all checks for the vehicle.

        Args:
            veh_id (str): Vehicle ID.
        """
        traci.vehicle.setSpeedMode(veh_id, 32)
        traci.vehicle.setLaneChangeMode(veh_id, 0)


def interpolate_control_command(control_command, obs_dict):
    """Interpolate the control command.

    Args:
        control_command (NDECommand): Control command.
        obs_dict (dict): Observation of the ego agent.

    Returns:
        NDECommand: Interpolated control command.
    """
    if control_command.command_type == CommandType.TRAJECTORY:
        # 1. interpolate the trajectory
        future_trajectory_array = np.array(control_command.future_trajectory)
        # 1.1 check first point of the trajectory, if it is not the current position, add it
        trajectory_array = np.vstack(
            (
                np.array(
                    [
                        obs_dict["ego"]["position"][0],
                        obs_dict["ego"]["position"][1],
                        obs_dict["ego"]["heading"],
                        obs_dict["ego"]["velocity"],
                        0,
                    ]
                ),
                future_trajectory_array,
            )
        )
        # 1.2 Start trajectory time at 0, using the command time resolution.
        trajectory_array[:, -1] = np.array(
            [i * control_command.time_resolution for i in range(len(trajectory_array))]
        )
        # 1.3 interpolate the trajectory
        dt = traci.simulation.getDeltaT()
        interpolated_trajecotory = interpolate_future_trajectory(trajectory_array, dt)
        # 1.4 update the time of the trajectory
        interpolated_trajecotory[:, -1] += traci.simulation.getTime()
        control_command.future_trajectory = interpolated_trajecotory[
            1:
        ]  # TODO: Angle cannot be interpolated
        # 1.5 update the time resolution of the trajectory
        control_command.time_resolution = dt
        # 1.6. clip the trajectory based on the control command duration
        control_command.future_trajectory = control_command.future_trajectory[
            : int(control_command.duration / control_command.time_resolution)
        ]
        return control_command
    else:
        return control_command
