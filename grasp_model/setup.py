from setuptools import find_packages, setup

package_name = 'grasp_model'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='farhan',
    maintainer_email='farhan@todo.todo',
    description='Generating grasps',
    license='Apache - 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'grconv_service = grasp_model.grconv_service:main',
        'generate_3d_grasp_pose = grasp_model.generate_3d_grasp_pose:main'
        ],
    },
)
