from glob import glob
import os

from setuptools import find_packages, setup

package_name = "warehouse_fleet"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    package_data={"warehouse_fleet": ["web/*"]},
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (
            os.path.join("share", package_name, "config"),
            ["../../../config/warehouse.yaml"],
        ),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*")),
        (os.path.join("share", package_name, "web"), glob("warehouse_fleet/web/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Arun",
    maintainer_email="arun@example.com",
    description="Autonomous multi-robot fleet control for the warehouse demo",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fleet_manager = warehouse_fleet.fleet_manager:main",
            "amr_agent = warehouse_fleet.amr_agent:main",
            "franka_arm = warehouse_fleet.franka_arm:main",
            "humanoid_patrol = warehouse_fleet.humanoid_patrol:main",
            "fleet_monitor = warehouse_fleet.fleet_monitor:main",
            "submit_task = warehouse_fleet.submit_task:main",
            "teleop_amr = warehouse_fleet.teleop_amr:main",
            "dashboard_server = warehouse_fleet.dashboard_server:main",
        ],
    },
)
