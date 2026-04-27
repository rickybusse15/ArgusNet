"""
Integration example: Complete workflow with terrain-aware zones and rejection diagnostics.

This script demonstrates:
1. Creating an environment with terrain
2. Generating mission zones with terrain-aware heights
3. Running tracking with full rejection diagnostics
4. Examining rejection geometry for visualization
"""

from __future__ import annotations

import numpy as np

from argusnet.adapters.argusnet_grpc import TrackerConfig, TrackingService
from argusnet.core.types import (
    ZONE_TYPE_EXCLUSION,
    ZONE_TYPE_OBJECTIVE,
    ZONE_TYPE_SURVEILLANCE,
    BearingObservation,
    MissionZone,
    NodeState,
    TruthState,
    vec3,
)
from argusnet.world.environment import (
    Bounds2D,
    CylinderObstacle,
    EnvironmentCRS,
    EnvironmentModel,
    LandCoverLayer,
    ObstacleLayer,
    TerrainLayer,
)


def create_terrain_environment() -> EnvironmentModel:
    """Create a realistic environment with terrain and obstacles."""
    bounds = Bounds2D(x_min_m=0.0, x_max_m=1000.0, y_min_m=0.0, y_max_m=1000.0)

    # Create rolling terrain
    x_vals = np.linspace(0, 1000, 20)
    y_vals = np.linspace(0, 1000, 20)
    heights = np.zeros((len(y_vals), len(x_vals)))

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            # Gentle rolling hills with a valley
            height = 50.0 + 20.0 * np.sin(x / 200.0) + 15.0 * np.cos(y / 150.0)
            heights[i, j] = height

    terrain = TerrainLayer.from_height_grid(
        environment_id="rolling-hills",
        bounds_xy_m=bounds,
        heights_m=heights,
        resolution_m=50.0,
    )

    # Add obstacles (buildings, walls)
    obstacles_list = [
        CylinderObstacle(
            primitive_id="building-1",
            blocker_type="building",
            center_x_m=250.0,
            center_y_m=250.0,
            radius_m=40.0,
            base_z_m=terrain.height_at(250.0, 250.0),
            top_z_m=terrain.height_at(250.0, 250.0) + 30.0,
        ),
        CylinderObstacle(
            primitive_id="building-2",
            blocker_type="building",
            center_x_m=750.0,
            center_y_m=250.0,
            radius_m=50.0,
            base_z_m=terrain.height_at(750.0, 250.0),
            top_z_m=terrain.height_at(750.0, 250.0) + 35.0,
        ),
        CylinderObstacle(
            primitive_id="tower",
            blocker_type="building",
            center_x_m=500.0,
            center_y_m=500.0,
            radius_m=20.0,
            base_z_m=terrain.height_at(500.0, 500.0),
            top_z_m=terrain.height_at(500.0, 500.0) + 50.0,  # Tall structure
        ),
    ]

    obstacles = ObstacleLayer(bounds_xy_m=bounds, tile_size_m=500.0, primitives=obstacles_list)

    land_cover = LandCoverLayer.open_terrain(bounds_xy_m=bounds, resolution_m=50.0)

    return EnvironmentModel(
        environment_id="example-environment",
        crs=EnvironmentCRS(),
        bounds_xy_m=bounds,
        terrain=terrain,
        obstacles=obstacles,
        land_cover=land_cover,
    )


def create_mission_zones(env: EnvironmentModel) -> list[MissionZone]:
    """Create mission zones with terrain-aware heights."""
    zones = []

    # Surveillance zone at elevated location
    surv_x, surv_y = 200.0, 200.0
    surv_height = env.terrain.height_at(surv_x, surv_y)
    zones.append(
        MissionZone(
            zone_id="surveillance-north",
            zone_type=ZONE_TYPE_SURVEILLANCE,
            center=vec3(surv_x, surv_y, surv_height),
            radius_m=150.0,
            priority=1,
            label="perimeter-surveillance",
        )
    )

    # Exclusion zone around tall tower
    excl_x, excl_y = 500.0, 500.0
    excl_height = env.terrain.height_at(excl_x, excl_y)
    zones.append(
        MissionZone(
            zone_id="exclusion-tower",
            zone_type=ZONE_TYPE_EXCLUSION,
            center=vec3(excl_x, excl_y, excl_height),
            radius_m=100.0,
            priority=2,
            label="keep-distance",
        )
    )

    # Objective zone at lower elevation
    obj_x, obj_y = 800.0, 800.0
    obj_height = env.terrain.height_at(obj_x, obj_y)
    zones.append(
        MissionZone(
            zone_id="objective-south",
            zone_type=ZONE_TYPE_OBJECTIVE,
            center=vec3(obj_x, obj_y, obj_height),
            radius_m=75.0,
            priority=3,
            label="primary-objective",
        )
    )

    return zones


def run_tracking_scenario(env: EnvironmentModel) -> None:
    """Run a tracking scenario and examine rejections."""
    platform = TrackingService(config=TrackerConfig())

    # Set up sensors at different elevations
    nodes = [
        NodeState(
            "sensor-north",
            vec3(100.0, 100.0, env.terrain.height_at(100.0, 100.0) + 20.0),
            vec3(0.0, 0.0, 0.0),
            False,
            0.0,
        ),
        NodeState(
            "sensor-south",
            vec3(900.0, 900.0, env.terrain.height_at(900.0, 900.0) + 20.0),
            vec3(0.0, 0.0, 0.0),
            False,
            0.0,
        ),
    ]

    # Targets at various locations
    truths = [
        TruthState(
            "target-1",
            vec3(250.0, 250.0, env.terrain.height_at(250.0, 250.0) + 5.0),
            vec3(1.0, 0.0, 0.0),
            0.0,
        ),
        TruthState(
            "target-2",
            vec3(500.0, 500.0, env.terrain.height_at(500.0, 500.0) + 5.0),
            vec3(0.0, 1.0, 0.0),
            0.0,
        ),
    ]

    # Create observations (some will be rejected)
    observations = []

    # Good observation
    direction = truths[0].position - nodes[0].position
    direction = direction / np.linalg.norm(direction)
    observations.append(
        BearingObservation(
            node_id="sensor-north",
            target_id="target-1",
            origin=nodes[0].position,
            direction=direction,
            bearing_std_rad=0.01,
            timestamp_s=0.0,
            confidence=0.95,
        )
    )

    # Low-confidence observation (will be rejected)
    direction = truths[1].position - nodes[1].position
    direction = direction / np.linalg.norm(direction)
    observations.append(
        BearingObservation(
            node_id="sensor-south",
            target_id="target-2",
            origin=nodes[1].position,
            direction=direction,
            bearing_std_rad=0.01,
            timestamp_s=0.0,
            confidence=0.05,  # Below threshold
        )
    )

    # Unknown node (will be rejected)
    direction = truths[0].position - nodes[0].position
    direction = direction / np.linalg.norm(direction)
    observations.append(
        BearingObservation(
            node_id="ghost-sensor",  # Unknown
            target_id="target-1",
            origin=nodes[0].position,
            direction=direction,
            bearing_std_rad=0.01,
            timestamp_s=0.0,
            confidence=0.9,
        )
    )

    # Ingest frame
    frame = platform.ingest_frame(
        timestamp_s=0.0,
        node_states=nodes,
        observations=observations,
        truths=truths,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("TRACKING SCENARIO RESULTS")
    print("=" * 80)

    print(f"\nEnvironment: {env.environment_id}")
    print(
        f"Bounds: {env.bounds_xy_m.x_min_m:.0f}-{env.bounds_xy_m.x_max_m:.0f}m x "
        f"{env.bounds_xy_m.y_min_m:.0f}-{env.bounds_xy_m.y_max_m:.0f}m"
    )

    print(f"\nSensors: {len(frame.nodes)}")
    for node in frame.nodes:
        terrain_height = env.terrain.height_at(node.position[0], node.position[1])
        agl = node.position[2] - terrain_height
        print(f"  {node.node_id}: ({node.position[0]:.0f}, {node.position[1]:.0f})")
        print(f"    Elevation: {node.position[2]:.1f}m AGL: {agl:.1f}m")

    print(f"\nTargets (truths): {len(frame.truths)}")
    for truth in frame.truths:
        terrain_height = env.terrain.height_at(truth.position[0], truth.position[1])
        agl = truth.position[2] - terrain_height
        print(f"  {truth.target_id}: ({truth.position[0]:.0f}, {truth.position[1]:.0f})")
        print(f"    Elevation: {truth.position[2]:.1f}m AGL: {agl:.1f}m")

    print(f"\nTracks: {len(frame.tracks)}")
    for track in frame.tracks:
        print(f"  {track.track_id}: position={track.position}, stale={track.stale_steps}")

    print("\nMetrics:")
    print(f"  Accepted observations: {frame.metrics.accepted_observation_count}")
    print(f"  Rejected observations: {frame.metrics.rejected_observation_count}")

    print("\nRejection counts:")
    for reason, count in frame.metrics.rejection_counts.items():
        print(f"  {reason}: {count}")

    print(f"\nRejected observations: {len(frame.rejected_observations)}")
    for i, rejection in enumerate(frame.rejected_observations):
        print(f"\n  Rejection #{i + 1}:")
        print(f"    {rejection.node_id} -> {rejection.target_id}")
        print(f"    Reason: {rejection.reason}")
        if rejection.detail:
            print(f"    Detail: {rejection.detail}")
        if rejection.origin is not None:
            print(
                "    Origin: "
                f"({rejection.origin[0]:.1f}, {rejection.origin[1]:.1f}, "
                f"{rejection.origin[2]:.1f})"
            )
        if rejection.blocker_type:
            print(f"    Blocker type: {rejection.blocker_type}")
        if rejection.first_hit_range_m is not None:
            print(f"    Range to blocker: {rejection.first_hit_range_m:.1f}m")

    print(f"\nGeneration rejections: {len(frame.generation_rejections)}")

    print("\n" + "=" * 80)


def main() -> None:
    """Run complete integration example."""
    print("Creating terrain environment...")
    env = create_terrain_environment()

    print("Creating mission zones with terrain-aware heights...")
    zones = create_mission_zones(env)

    print("\nMission zones:")
    for zone in zones:
        print(f"  {zone.zone_id} ({zone.zone_type})")
        print(f"    Center: ({zone.center[0]:.0f}, {zone.center[1]:.0f}, {zone.center[2]:.1f}m)")
        print(f"    Radius: {zone.radius_m}m")

    print("\nRunning tracking scenario with rejection diagnostics...")
    run_tracking_scenario(env)

    print("\nIntegration example complete!")


if __name__ == "__main__":
    main()
