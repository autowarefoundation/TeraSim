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
    # Exact rear-axle reference corresponding to the front-centre target.
    control_lookahead_x: float = 0.0
    control_lookahead_y: float = 0.0
    control_lookahead_z: float = 0.0
    control_lookahead_position_valid: bool = False
    # Speed-adaptive rear target retained for continuity and ahead diagnostics.
    steering_preview_x: float = 0.0
    steering_preview_y: float = 0.0
    steering_preview_z: float = 0.0
    steering_preview_position_valid: bool = False
    steering_preview_requested_distance: float | None = None
    steering_preview_effective_distance: float | None = None
    steering_preview_front_s: float | None = None
    steering_preview_front_d: float | None = None
    steering_preview_tangent_x: float | None = None
    steering_preview_tangent_y: float | None = None
    steering_preview_normal_x: float | None = None
    steering_preview_normal_y: float | None = None
    front_rear_distance_error: float | None = None
    front_rear_distance_consistent: bool = False
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
    feedback_lateral_reference_preserved: bool = False
    feedback_preserved_sumo_lateral_offset: float | None = None
    feedback_assimilated_x: float | None = None
    feedback_assimilated_y: float | None = None

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
    av_control_mode: str = "external"
    av_lane_change_decision: dict = {}
    av_lane_change_request: dict = {}
    effective_speed_mode: int | None = None
    nde_lane_change_decision_source: str = ""
    nde_lane_change_command: str = "none"
    nde_lane_change_request_count: int = 0
    nde_lane_change_request_time: float | None = None
    nde_lane_change_source_lane_id: str = ""
    nde_lane_change_target_lane_id: str = ""
    nde_lane_change_duration: float | None = None
    nde_lane_change_model_state: int | None = None
    nde_lane_change_post_state: int | None = None
    nde_controller_busy: bool = False
    nde_command_end_reason: str = ""
    effective_lane_change_mode: int | None = None
    sumo_route: tuple[str, ...] = ()
    # SUMO odometer and route occurrence disambiguate loop routes whose edge
    # IDs repeat. They are the authoritative closed-loop progress record.
    sumo_route_index: int | None = None
    sumo_route_distance: float | None = None
    external_state_lateral_maneuver_enabled: bool = False
    external_state_lateral_maneuver_target_enabled: bool = False
    external_state_maneuver_source_lane_id: str = ""
    external_state_maneuver_current_lane_id: str = ""
    external_state_maneuver_target_lane_id: str = ""
    external_state_maneuver_phase: str = "lane_keep"
    external_state_maneuver_armed_origin_phase: str = "lane_keep"
    external_state_maneuver_target_pre_state_phase: str = "lane_keep"
    external_state_maneuver_target_post_state_phase: str = "lane_keep"
    external_state_maneuver_target_applied: bool = False
    external_state_maneuver_target_source: str = "route_center"
    external_state_maneuver_target_selected_d: float | None = None
    external_state_maneuver_target_world_anchor_x: float | None = None
    external_state_maneuver_target_world_anchor_y: float | None = None
    external_state_maneuver_target_d_outside_lane: bool | None = None
    external_state_maneuver_target_d_outside_primary_lane: bool | None = None
    external_state_maneuver_target_safety_region: str = "primary_lane"
    external_state_active_maneuver_corridor_valid: bool | None = None
    external_state_active_maneuver_corridor_reason: str = ""
    external_state_active_maneuver_corridor_d_min: float | None = None
    external_state_active_maneuver_corridor_d_max: float | None = None
    external_state_active_maneuver_corridor_source_center_d: float | None = None
    external_state_active_maneuver_corridor_target_center_d: float | None = None
    external_state_active_maneuver_corridor_center_separation: float | None = None
    external_state_maneuver_primary_lane_switched: bool = False
    external_state_maneuver_current_lateral_offset: float | None = None
    external_state_maneuver_preview_lateral_offset: float | None = None
    external_state_maneuver_held_lateral_offset: float | None = None
    external_state_maneuver_active_frame_count: int = 0
    external_state_maneuver_zero_motion_frame_count: int = 0
    external_state_maneuver_transition_reason: str = ""
    external_state_maneuver_evidence_eligible: bool = False
    external_state_maneuver_evidence_exclusion_reason: str = ""
    external_state_maneuver_canonical_motion_evidence: bool = False
    external_state_maneuver_canonical_delta_sign: int = 0
    external_state_maneuver_speed_motion_evidence: bool = False
    external_state_maneuver_intent_evidence: bool = False
    external_state_maneuver_authorized: bool = False
    external_state_maneuver_authorization_source: str = "none"
    external_state_maneuver_authorized_direction: str = "none"
    external_state_maneuver_start_candidate_sign: int = 0
    external_state_maneuver_start_candidate_frame_count: int = 0
    external_state_maneuver_armed_idle_frame_count: int = 0
    external_state_maneuver_confirmed_direction_sign: int = 0
    external_state_maneuver_confirmed_sumo_time: float | None = None
    external_state_maneuver_confirmed_direction_x: float | None = None
    external_state_maneuver_confirmed_direction_y: float | None = None
    external_state_maneuver_switch_evidence_excluded: bool = False
    feedback_observed_x: float | None = None
    feedback_observed_y: float | None = None
    feedback_observed_sumo_angle: float | None = None
    feedback_requested_sumo_angle: float | None = None
    feedback_observed_speed: float | None = None
    feedback_observed_acceleration: float | None = None
    feedback_rear_axle_x: float | None = None
    feedback_rear_axle_y: float | None = None
    front_to_rear_axle_distance: float | None = None
    feedback_requested_lane_id: str | None = None
    feedback_observed_lane_id: str | None = None
    feedback_phase_a_sumo_time: float | None = None
    feedback_source_carla_frame: int | None = None
    feedback_longitudinal_error: float | None = None

    # Canonical route-Frenet diagnostics.
    canonical_route_lane_ids: tuple[str, ...] = ()
    canonical_route_path_key: tuple[str, ...] = ()
    canonical_route_occurrence_ids: tuple[int, ...] = ()
    canonical_rolling_history_lane_ids: tuple[str, ...] = ()
    canonical_rolling_history_distance: float | None = None
    canonical_rolling_retained_history_distance: float | None = None
    canonical_route_path_switched: bool = False
    canonical_phase_a_s: float | None = None
    canonical_phase_a_d: float | None = None
    canonical_evidence_phase_a_s: float | None = None
    canonical_evidence_phase_a_d: float | None = None
    canonical_phase_b_s: float | None = None
    canonical_phase_a_projection_distance: float | None = None
    canonical_phase_a_projection_along_residual: float | None = None
    canonical_phase_a_projection_endpoint_clamped: bool = False
    canonical_phase_b_d: float | None = None
    canonical_delta_d: float | None = None
    phase_a_to_b_delta_x: float | None = None
    phase_a_to_b_delta_y: float | None = None
    phase_a_to_b_velocity_x: float | None = None
    phase_a_to_b_velocity_y: float | None = None
    phase_a_to_b_requested_longitudinal_speed: float | None = None
    phase_a_to_b_requested_lateral_speed: float | None = None
    phase_a_to_b_motion_heading_radians: float | None = None
    phase_a_to_b_motion_heading_degrees: float | None = None
    phase_a_to_b_motion_heading_valid: bool = False
    phase_a_to_b_motion_heading_invalid_reason: str = ""
    phase_a_to_b_lateral_motion_requested: bool = False
    canonical_phase_b_projection_distance: float | None = None
    canonical_phase_b_projection_along_residual: float | None = None
    canonical_phase_b_projection_endpoint_clamped: bool = False
    canonical_target_s: float | None = None
    canonical_target_d: float | None = None
    path_tracker_valid: bool = False
    path_tracker_center_curvature: float | None = None
    canonical_curvature_half_window: float | None = None
    path_tracker_front_curvature: float | None = None
    path_tracker_rear_curvature: float | None = None
    path_tracker_reference_tangent_x: float | None = None
    path_tracker_reference_tangent_y: float | None = None
    path_tracker_front_cross_track_error: float | None = None
    path_tracker_motion_heading_applied: bool = False
    path_tracker_reference_heading_source: str = "route_tangent"
    path_tracker_motion_heading_reason: str = ""
    canonical_held_d: float | None = None
    canonical_hold_reprojected: bool = False
    canonical_invariant_valid: bool = True
    canonical_invariant_reason: str = ""
    canonical_invariant_warning: str = ""
    canonical_fallback_applied: bool = False
    canonical_fallback_reason: str = ""
    canonical_fallback_age: int = 0
    canonical_front_target_lateral_jump: float | None = None
    canonical_control_target_lateral_jump: float | None = None
    canonical_steering_preview_lateral_jump: float | None = None
    canonical_front_target_lateral_change: float | None = None
    canonical_control_target_lateral_change: float | None = None
    canonical_steering_preview_lateral_change: float | None = None
    canonical_current_d_outside_lane: bool | None = None
    canonical_held_d_outside_lane: bool | None = None

    # Orientation
    orientation: float = 0.0
    acceleration: float = 0.0
    angular_velocity: float = 0.0

    # additional information of the agent
    type: str = ""
