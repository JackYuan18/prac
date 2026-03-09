# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 23:56:55 2026

@author: yzyja
"""

import pybullet as p
import time
import pybullet_data # Helper to find data files

# 1. Connect to the physics server (GUI version for visualization)
if p.isConnected():
    p.disconnect()
physicsClient = p.connect(p.GUI) 

# 2. Optionally, set the path to the pybullet_data to load standard objects
p.setAdditionalSearchPath(pybullet_data.getDataPath()) 

# 3. Set gravity
p.setGravity(0, 0, -10) 

# 4. Load a plane URDF (ground) and a simple object (e.g., R2D2 robot)
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 1] # x, y, z position
startOrientation = p.getQuaternionFromEuler([0, 0, 0]) # Euler angles converted to quaternion
boxId = p.loadURDF("pendulum5.urdf", startPos, startOrientation)

# 5. Run the simulation loop
for i in range(1000):
    p.stepSimulation() # Advance the simulation by one step
    time.sleep(1./240.) # Optional: add a small delay to control visualization speed

# 6. Disconnect after the loop (or close the window)
p.disconnect()
