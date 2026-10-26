"""Opt-in control profiles for RoboDojo policy pilots."""


HYBRID = "hybrid"
QUEUED = "queued"
LEGACY_CONTINUOUS_FAST = "continuous_fast"


def apply_control_profile(config, profile, *, task, reference_horizon):
    """Apply an explicitly named execution profile without changing the baseline."""
    if profile == "standard":
        return {
            "name": profile,
            "reference_horizon": reference_horizon,
            "action_horizon": config["predict_horizon"],
            "plan_continuity": False,
        }
    if profile not in (HYBRID, LEGACY_CONTINUOUS_FAST, QUEUED):
        raise ValueError(f"Unsupported control profile: {profile}")
    if task != "put_bottles_into_dustbin":
        raise ValueError(f"{profile} is currently piloted only for put_bottles_into_dustbin")

    # Keep the public RoboICL retrieval chunks at their recorded horizon. The
    # LIVE policy may emit a longer, independently validated action sequence.
    config.update(
        predict_horizon=32,
        execute_horizon=32,
        fixed_predict_horizon=False,
        live_anchor_count=2,
        live_retain_latest=True,
        # In hybrid mode a submitted row is a sparse waypoint displacement.
        # The policy-side motor adapter subdivides it before dispatch.
        max_translation_m=0.30,
        max_rotation_rad=0.60,
    )
    metadata = {
        "name": profile,
        "reference_horizon": reference_horizon,
        "action_horizon": 32,
        "plan_continuity": True,
        "control_modes": ["waypoint_transit", "high_frequency_precision"],
        "transit_action_horizon": 32,
        "precision_action_horizon": [4, 8],
        "precision_transition_distance_m": 0.15,
        "sparse_route_anchor_count": [3, 6],
        "waypoint_translation_limit_m": config["max_translation_m"],
        "execution_step_translation_limit_m": 0.03,
        "execution_step_rotation_limit_rad": 0.15,
        "live_anchor_count": 2,
    }
    if profile == QUEUED:
        # Legacy prompt hints are not enforced transition rules in this pilot.
        for key in ("transit_action_horizon", "precision_transition_distance_m",
                    "sparse_route_anchor_count"):
            metadata.pop(key)
        metadata.update(
            executor_version="queued-v1",
            plan_continuity=False,  # Semantic task planning is not implemented.
            motion_queue_persistent=True,
            queue_capacity_steps=64,
            dispatch_window_steps=32,
            control_modes=["transit", "precision"],
            transit_waypoint_count=[1, 6],
            precision_action_horizon=[1, 8],
            precision_translation_limit_m=0.01,
            enforced_transit_gripper_hold=True,
            phase_authorization=False,
            execution_mode="synchronous",
        )
    return metadata
