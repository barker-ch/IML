from setuptools import find_packages, setup

package_name = 'vn_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/vn_driver_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cbarker',
    maintainer_email='barker.ch@northeastern.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vn_driver = vn_driver.vn_driver:main',
        ],
    },
)
