# Instructions to run the repository:
   1) Navigate to your workspace's src directory:
      bash
   cd ~/your_workspace/src
   2) Clone the repository:
      bash
      git clone https://github.com/salochinYom/VBM_PR1_workspace_M2.git
   3) Build the workspace
      bash 
      colcon build --symlink install
   4) Build the workspace
      bash 
      source install/setup.bash
   5) Run the simulation:
      bash 
      ros2 launch vbm_project_env simulation.launch.py
   6) Start the grasp model service:
      bash
      ros2 run grasp_model grconv_service
   7) Call this service to run GRCONVNet:
      bash
      ros2 service call /grconv_model define_service/srv/GrConv "{model: 'use_grconvnet'}"
   8) Call this service to run GGCNN:
      bash
      ros2 service call /grconv_model define_service/srv/GrConv "{model: 'use_ggcnn'}"
   9) Generate 3D grasps:
      bash
      ros2 run grasp_model generate_3d_grasp
    
   9) Visualize in RViz:
      bash
      ros2 run rviz2 rviz2

