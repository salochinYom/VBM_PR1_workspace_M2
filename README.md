# Instructions to run the repository:

### 1) Navigate to your workspace's `src` directory:

      cd ~/your_workspace/src 

###  2) Clone the repository:

      git clone https://github.com/salochinYom/VBM_PR1_workspace_M2.git

 ### 3) Build the workspace:

      colcon build --symlink install
      
 ### 4) Source the workspace:

      source install/setup.bash
      
 ### 5) Run the simulation:

      ros2 launch vbm_project_env simulation.launch.py
      
###  6) Start the grasp model service:
      
       ros2 run grasp_model grconv_service

###   7) Call this service to run GRCONVNet:
     
      ros2 service call /grconv_model define_service/srv/GrConv "{model: 'use_grconvnet'}"
      
###   8) Call this service to run GGCNN:
    
      ros2 service call /grconv_model define_service/srv/GrConv "{model: 'use_ggcnn'}"
      
 ###  9) Generate 3D grasps:

      ros2 run grasp_model generate_3d_grasp
    
  ### 10) Visualize in RViz:
 
      ros2 run rviz2 rviz2


