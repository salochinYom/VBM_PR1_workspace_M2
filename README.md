# Instructions to run the repository:
1) Navigate to your workspace's src directory:
   bash
   cd ~/your_workspace/src
2) Clone the repository:
   bash
   git clone https://github.com/salochinYom/VBM_PR1_workspace_M2.git
3) 

   colcon build --symlink-install
   
5) Launch the Gazebo:
   bash
   ros2 launch vbm_project_env simulation.launch.py
   
6) Run RVIZ:
   bash
   ros2 run rviz2 rviz2
   
7) Launch Grasp Pose Generator:
   bash
   ros2 launch grasp_pose_generator grasp_pose_generator_launch.py 
   
8) Start the Grasp Generation Service:
   bash
   ros2 run grasp_gen_service grasp_gen_service
   
9) Service call:
   bash
   ros2 service call /generate_grasp grasp_gen_interface/srv/GraspGen input:\ 'generate_grasp_grconvnet'\
   
   or
   bash
   ros2 service call /generate_grasp grasp_gen_interface/srv/GraspGen input:\ 'generate_grasp_ggcnn'\
   
   or
    bash
   ros2 service call /generate_grasp grasp_gen_interface/srv/GraspGen input:\ 'generate_grasp_ggcnn2'\
