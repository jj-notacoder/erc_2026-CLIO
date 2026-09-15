from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'erc_phase1_solution'


setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md', 'THIRD_PARTY_NOTICES.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (
            os.path.join('share', package_name, 'assets', 'digits'),
            glob(os.path.join('assets', 'digits', '*.png')),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ERC Phase 1 Team',
    maintainer_email='team@example.com',
    description=(
        'Autonomous perception, navigation, manipulation, and trial telemetry '
        'for ERC 2026 Phase 1.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = erc_phase1_solution.perception_node:main',
            'navigation_node = erc_phase1_solution.navigation_node:main',
            'manipulation_node = erc_phase1_solution.manipulation_node:main',
            'mission_manager = erc_phase1_solution.mission_manager:main',
        ],
    },
)
