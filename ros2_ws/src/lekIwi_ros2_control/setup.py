import os
from glob import glob

from setuptools import setup

package_name = "lekIwi_ros2_control"

# The executable Python nodes live in scripts/, which is exposed as the
# importable package `lekIwi_ros2_control` so that console_scripts entry
# points work while still keeping the requested file layout:
#     ros2_ws/src/lekIwi_ros2_control/scripts/lekIwi_teleop.py
setup(
    name=package_name,
    version="0.1.0",
    description="ROS 2 Humble control stack for the LeKiwi robot on Isaac Sim 5.1",
    author="arun",
    license="Apache-2.0",
    packages=[package_name],
    package_dir={package_name: "scripts"},
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="arun",
    maintainer_email="arun@example.com",
    entry_points={
        "console_scripts": [
            "lekIwi_base_controller = "
            "lekIwi_ros2_control.lekIwi_base_controller:main",
            "lekIwi_teleop = lekIwi_ros2_control.lekIwi_teleop:main",
            "test_arm_trajectory = "
            "lekIwi_ros2_control.test_arm_trajectory:main",
        ],
    },
)
