from pydantic import BaseModel


class AgentStateSimplified(BaseModel):
    # Position
    ## x position of the agent in the SUMO coordinate system (meters)
    x: float = 0.0
    ## y position of the agent in the SUMO coordinate system (meters)
    y: float = 0.0
    ## elevation of the agent (meters)
    z: float = 0.0

    # Lane-relative reconstructed position
    lane_id: str = ""
    lane_position: float = 0.0
    lateral_offset: float = 0.0
    reconstructed_x: float = 0.0
    reconstructed_y: float = 0.0
    reconstructed_z: float = 0.0
    reconstructed_position_valid: bool = False

    # Path-following lookahead target in the SUMO coordinate system.
    lookahead_x: float = 0.0
    lookahead_y: float = 0.0
    lookahead_z: float = 0.0
    lookahead_position_valid: bool = False
    lookahead_distance: float = 0.0
    lookahead_heading_change: float = 0.0
    lookahead_lane_change_blend: float = 0.0
    lookahead_route_x: float = 0.0
    lookahead_route_y: float = 0.0
    lookahead_route_z: float = 0.0
    lookahead_action_mode: str = "route"
    lookahead_action_valid: bool = True
    lookahead_action_error: str = ""
    lookahead_action_warning: str = ""
    lookahead_lateral_direction_source: str = "inactive"
    lookahead_lateral_direction_unresolved_count: int = 0
    lookahead_lane_change_intent_conflict: bool = False
    lookahead_lateral_horizon_displacement: float = 0.0
    lookahead_target_lateral_distance: float | None = None
    lookahead_route_tangent_x: float | None = None
    lookahead_route_tangent_y: float | None = None
    lookahead_world_left_normal_x: float | None = None
    lookahead_world_left_normal_y: float | None = None
    lookahead_phase_b_lateral_delta: float | None = None
    lookahead_expected_phase_b_lateral_distance: float | None = None
    lookahead_world_lateral_speed: float = 0.0
    lookahead_origin_x: float = 0.0
    lookahead_origin_y: float = 0.0
    lateral_speed: float = 0.0
    feedback_position_skipped_for_lane_change: bool = False

    ## longitude of the agent (degrees)
    lon: float = 0.0
    ## latitude of the agent (degrees)
    lat: float = 0.0

    # Orientation in the SUMO coordinate system
    sumo_angle: float = 0.0

    # Road-relative pitch reported by SUMO (degrees). Keep the native unit
    # because this value is passed directly to carla.Rotation.pitch.
    sumo_slope: float = 0.0

    # Size (https://www.autoscout24.de/auto/technische-daten/mercedes-benz/vito/vito-111-cdi-kompakt-2003-2014-transporter-diesel/)
    ## length of the agent (meters)
    length: float = 5.0
    ## width of the agent (meters)
    width: float = 1.8
    ## height of the agent (meters)
    height: float = 1.5

    # Speed
    speed: float = 0.0

    # SUMO's next desired speed and the latest CARLA observation accepted by SUMO.
    sumo_desired_speed: float | None = None
    sumo_emergency_decel: float | None = None
    sumo_lane_change_intent: str = "none"
    sumo_lane_change_target_lane_id: str = ""
    sumo_route: tuple[str, ...] = ()
    external_state_maneuver_source_lane_id: str = ""
    external_state_maneuver_target_lane_id: str = ""
    feedback_observed_x: float | None = None
    feedback_observed_y: float | None = None
    feedback_observed_sumo_angle: float | None = None
    feedback_observed_speed: float | None = None
    feedback_observed_acceleration: float | None = None
    feedback_requested_lane_id: str | None = None
    feedback_observed_lane_id: str | None = None
    feedback_phase_a_sumo_time: float | None = None
    feedback_source_carla_frame: int | None = None
    feedback_longitudinal_error: float | None = None

    # Orientation
    orientation: float = 0.0
    acceleration: float = 0.0
    angular_velocity: float = 0.0

    # additional information of the agent
    type: str = ""
