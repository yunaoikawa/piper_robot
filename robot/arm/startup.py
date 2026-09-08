"""Shared arm startup sequences for policy inference and teleoperation."""


def prepare_arms_for_manipulation(
    robot_rpc,
    *,
    arms=("left", "right"),
    context="startup",
):
    """Visit true machine zero, then the upright manipulation home.

    ConeE must already have been initialized with ``reset_arms=False`` so the
    machine-zero visit happens exactly once.  The default bimanual path uses a
    single RPC operation to preserve the server's deterministic left-then-right
    order.  Single-arm legacy teleop entrypoints use the matching arm-specific
    operation.
    """
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("arms must contain one or more unique arm names")
    unknown = set(arms) - {"left", "right"}
    if unknown:
        raise ValueError(f"unknown arm(s): {sorted(unknown)}")

    prefix = f"[{context}]"
    if arms == ("left", "right"):
        print(f"{prefix} Returning both arms to true machine zero (q=0)...",
              flush=True)
        robot_rpc.machine_zero_arms()
    else:
        for arm in arms:
            print(f"{prefix} Returning {arm} arm to true machine zero (q=0)...",
                  flush=True)
            getattr(robot_rpc, f"machine_zero_{arm}_arm")()

    for arm in arms:
        print(f"{prefix} Moving {arm} arm to manipulation home...", flush=True)
        getattr(robot_rpc, f"home_{arm}_arm")()
    print(f"{prefix} Arm initialization complete.", flush=True)
