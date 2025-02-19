# Check if ethtool is installed
if ! dpkg -l | grep -q "ethtool"; then
    echo "\e[31mError: ethtool not detected in the system.\e[0m"
    echo "Please use the following command to install ethtool:"
    echo "sudo apt update && sudo apt install ethtool"
    exit 1
fi

# Check if can-utils is installed
if ! dpkg -l | grep -q "can-utils"; then
    echo "\e[31mError: can-utils not detected in the system.\e[0m"
    echo "Please use the following command to install can-utils:"
    echo "sudo apt update && sudo apt install can-utils"
    exit 1
fi

echo "Both ethtool and can-utils are installed."

# Iterate over all CAN interfaces
for iface in $(ip -br link show type can | awk '{print $1}'); do
    # Use ethtool to get bus-info
    BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
    
    if [ -z "$BUS_INFO" ]; then
        echo "Error: Unable to retrieve bus-info information for interface $iface."
        continue
    fi
    
    echo "Interface $iface is connected to USB port $BUS_INFO"
done
