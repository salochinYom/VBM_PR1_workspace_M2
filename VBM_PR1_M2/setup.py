from setuptools import find_packages, setup

package_name = 'VBM_PR1_M2'

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
    maintainer='nmoy',
    maintainer_email='73899470+salochinYom@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'listner = VBM_PR1_M2.subscriber_member_function:main',
            'service = VBM_PR1_M2.servicecaller:main',
            'ggcnn_model_host = VBM_PR1_M2.ggcnn.ggcnn_grasp_processor:main'
        ],
    },
)
