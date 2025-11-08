
from setuptools import setup, find_packages

package_name = 'ltm_poc'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/ltm_poc']),
        ('share/ltm_poc', ['package.xml']),
        ('share/ltm_poc/launch', ['launch/ltm_poc_demo.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Minimal PoC: YOLO + CLIP + viewer + controller',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_detector_node = ltm_poc.simple_detector_node:main',
            'vlm_clip_node = ltm_poc.vlm_clip_node:main',
            'viewer_node = ltm_poc.viewer_node:main',
            'controller_node = ltm_poc.controller_node:main',
        ],
    },
)
