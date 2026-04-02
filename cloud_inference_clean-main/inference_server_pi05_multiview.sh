# Get the compute node's IP address
NODE_IP=$(hostname -i)
NODE_NAME=$(hostname)

# Create reverse SSH tunnel to your workstation
# Replace with your actual workstation address
WORKSTATION="216.165.70.11"
WORKSTATION_PORT_OBS=15555
WORKSTATION_PORT_ACT=15556
WORKSTATION_PORT_CTRL=15557
WORKSTATION_USER="yz12129"

echo "Setting up reverse tunnel to ${WORKSTATION}"

# Establish reverse tunnel (requires your workstation to accept SSH)
ssh -f -N -R ${WORKSTATION_PORT_OBS}:localhost:8555 \
           -R ${WORKSTATION_PORT_ACT}:localhost:5556 \
              -R ${WORKSTATION_PORT_CTRL}:localhost:5557 \
                ${WORKSTATION_USER}@${WORKSTATION}

echo "=========================================="
echo "Inference server starting on:"
echo "  Node: $NODE_NAME"
echo "  IP: $NODE_IP"
echo "  Observation port: 8555"
echo "  Action port: 5556"
echo "=========================================="
echo ""
# Save tunnel info
echo "Connect from workstation using:"
echo "  python cloud_inference_control_collect_v2.py --record --host localhost \\"
echo "    --obs-port ${WORKSTATION_PORT_OBS} --action-port ${WORKSTATION_PORT_ACT}"
# echo "${WORKSTATION_PORT_OBS}" > ~/robot_inference/tunnel_obs_port.txt
# echo "${WORKSTATION_PORT_ACT}" > ~/robot_inference/tunnel_act_port.txt

# Write connection info to file for easy access
# mkdir -p ~/robot_inference
# echo "$NODE_IP" > ~/robot_inference/current_node_ip.txt
# echo "$NODE_NAME" > ~/robot_inference/current_node_name.txt
# date > ~/robot_inference/last_started.txt

# Run the asynchronous inference server
PYTHONUNBUFFERED=1 python hpc_inference_pi05.py \
    --checkpoint "path_to_model" \
    --obs-port 8555 \
    --action-port 5556 \
    --device cuda \
    --pred_horizon 50 \
    --temp_ensemble

echo ""
echo "Inference server stopped at $(date)"