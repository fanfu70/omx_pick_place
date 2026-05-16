from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'omx_pick_place'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Vision-guided pick and place for OpenManipulator-X',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'calibration_node = omx_pick_place.calibration_node:main',
            'color_detector = omx_pick_place.color_detector:main',
            'depth_localizer = omx_pick_place.depth_localizer:main',
            'pick_place_node = omx_pick_place.pick_place_node:main',
        ],
    },
)
