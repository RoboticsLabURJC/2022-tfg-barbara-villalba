#!/usr/bin/env python3 

import airsim

client = airsim.MultirotorClient()
client.confirmConnection()

for vehicles in client.listVehicles():
    
    client.enableApiControl(True,vehicles)
    

client.takeoffAsync(2,'drone2')


