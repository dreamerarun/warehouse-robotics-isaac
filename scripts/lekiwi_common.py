# SPDX-License-Identifier: Apache-2.0
"""Shared USD helpers for the LeKiwi Isaac Sim scripts.

Thin, defensive wrappers around the ``pxr.UsdPhysics`` schemas so the import
script stays readable and every step is reported instead of silently ignored.
"""

from pxr import Sdf, UsdPhysics


def configure_drives(joint_prim, target_type, target, stiffness, damping, max_force):
    """Author ``physics:drive:angular`` on a revolute joint.

    Args:
        joint_prim (Usd.Prim): the joint prim, e.g. ``/World/LeKiwi/joints/<name>``.
        target_type (str): ``"position"`` or ``"velocity"``.
        target (float): initial target, in radians (converted to degrees USD expects).
        stiffness (float): spring stiffness. Use 0 for a pure velocity servo.
        damping (float): damping coefficient.
        max_force (float): actuator force/torque cap.
    """
    if target_type not in ("position", "velocity"):
        raise ValueError("target_type must be 'position' or 'velocity'")

    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")

    if target_type == "position":
        attr = drive.GetTargetPositionAttr()
        if attr:
            attr.Set(float(target))
        else:
            drive.CreateTargetPositionAttr(float(target))
        # a velocity servo must not fight a position spring
        va = drive.GetTargetVelocityAttr()
        if va:
            va.Set(0.0)
    else:
        attr = drive.GetTargetVelocityAttr()
        if attr:
            attr.Set(float(target))
        else:
            drive.CreateTargetVelocityAttr(float(target))
        pa = drive.GetTargetPositionAttr()
        if pa:
            pa.Set(0.0)

    for name, value in (("stiffness", stiffness), ("damping", damping), ("max_force", max_force)):
        if value is None:
            continue
        attr = getattr(drive, "Get{}Attr".format(name.capitalize()))()
        if attr:
            attr.Set(float(value))
        else:
            getattr(drive, "Create{}Attr".format(name.capitalize()))(float(value))


def configure_limits(joint_paths, arm_limits_deg, wheel_limits_deg):
    """Write ``physics:lowerLimit`` / ``physics:upperLimit`` in DEGREES.

    The source URDF declares every joint as ``continuous`` with no ``<limit>``
    element, so PhysX would otherwise inherit whatever the importer guessed.
    Wheels get a huge range so they can spin both ways; the arm gets the same
    conservative bounds the ROS 2 ``<ros2_control>`` block declares in radians.
    """
    report = []
    for name, prim in joint_paths.items():
        if name in arm_limits_deg:
            lo, hi = arm_limits_deg[name]
        else:
            lo, hi = wheel_limits_deg
        for attr_name, value in (("physics:lowerLimit", lo), ("physics:upperLimit", hi)):
            attr = prim.GetAttribute(attr_name)
            if not attr or not attr.IsValid():
                attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
            attr.Set(float(value))
        report.append((name, lo, hi))
    return report


def report_joints(stage, joint_paths):
    """Print a per-joint summary (type, drive gains, limits) for the log."""
    for name, prim in joint_paths.items():
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        lo = prim.GetAttribute("physics:lowerLimit")
        hi = prim.GetAttribute("physics:upperLimit")
        kind = "velocity" if (drive and drive.GetStiffnessAttr() and
                              drive.GetStiffnessAttr().Get() == 0) else "position"
        print(
            "[import_lekIwi]   joint {:40s} type={:22s} drive={} "
            "stiff={:<8g} damp={:<8g} maxF={:<8g} limits=[{:.1f}, {:.1f}]deg".format(
                name,
                prim.GetTypeName(),
                kind,
                drive.GetStiffnessAttr().Get() if drive and drive.GetStiffnessAttr() else float("nan"),
                drive.GetDampingAttr().Get() if drive and drive.GetDampingAttr() else float("nan"),
                drive.GetMaxForceAttr().Get() if drive and drive.GetMaxForceAttr() else float("nan"),
                lo.Get() if lo else float("nan"),
                hi.Get() if hi else float("nan"),
            ),
            flush=True,
        )
