#!/bin/bash

# Instructions:
#
# 1. Prerequisites
#    You need to have the ip tool and ethtool tool installed.
#    sudo apt install ethtool can-utils
#    Make sure the gs_usb driver is properly installed.
#
# 2. Background
#    This script is designed to automatically manage, rename, and bring up CAN (Controller Area Network) interfaces.
#    It checks the current number of CAN modules in the system and renames and activates CAN interfaces based on predefined USB ports.
#    This is especially useful for a system with multiple CAN modules where each different module may require a specific name.
#
# 3. Main functions
#    - Check the number of CAN modules: Ensures that the number of detected CAN modules in the system matches the predefined number.
#    - Retrieve USB port information: Uses ethtool to obtain the USB port info for each CAN module.
#    - Validate USB ports: Checks whether each CAN module's USB port matches the predefined list of ports.
#    - Rename CAN interfaces: Renames the CAN interfaces to target names based on the predefined USB ports.
#
# 4. Script configuration
#    The key configuration items in the script include the expected CAN module count, the default CAN interface name, and the bitrate settings:
#    1. Expected number of CAN modules:
#       EXPECTED_CAN_COUNT=1
#       This value determines how many CAN modules should be detected in the system.
#    2. Default CAN interface name when there is only one CAN module:
#       DEFAULT_CAN_NAME="${1:-can0}"
#       You can specify the default CAN interface name via a command-line argument; if not provided, it defaults to can0.
#    3. Default bitrate when there is only one CAN module:
#       DEFAULT_BITRATE="${2:-500000}"
#       You can specify the bitrate for a single CAN module via a command-line argument; if not provided, it defaults to 500000.
#    4. Configuration for multiple CAN modules:
#       declare -A USB_PORTS
#       USB_PORTS["1-2:1.0"]="can_device_1:500000"
#       USB_PORTS["1-3:1.0"]="can_device_2:250000"
#       The key represents the USB port, and the value is the interface name and bitrate, separated by a colon.
#
# 5. Usage steps
#    1. Edit the script:
#       1. Modify predefined values:
#          - The predefined CAN module count: EXPECTED_CAN_COUNT=2 (change it to match how many CAN modules are inserted)
#          - If you only have one CAN module, set the above parameter and skip the rest of these multiple-module steps.
#          - (Multiple CAN modules) The predefined USB ports and target interface names:
#             First, insert one CAN module into the expected USB port (during the initial setup, insert one CAN module at a time).
#             Then run sudo ethtool -i can0 | grep bus, record the parameter that follows bus-info:
#             Then insert the next CAN module into another USB port (different from the first) and repeat the command above.
#             (In fact, you can use one CAN module to test multiple USB ports, since modules are differentiated by USB address.)
#             After you finish planning out which module goes into which USB port and have recorded the bus-info value,
#             modify the USB_PORTS (bus-info) and the target interface names accordingly.
#             For example:
#                can_device_1:500000 => the first is the chosen CAN interface name, and the second is the bitrate
#                declare -A USB_PORTS
#                USB_PORTS["1-2:1.0"]="can_device_1:500000"
#                USB_PORTS["1-3:1.0"]="can_device_2:250000"
#             You only need to change the inside of USB_PORTS["1-3:1.0"] to match what you recorded from bus-info:
#       2. Give the script execution permissions:
#          Open a terminal, navigate to the directory containing the script, and run:
#            chmod +x can_config.sh
#       3. Run the script:
#          Use sudo to execute the script, because it requires administrative privileges to modify network interfaces.
#          1. Single CAN module
#             1. You can specify the default CAN interface name and bitrate via command-line arguments (defaults are can0 and 500000):
#                sudo bash ./can_config.sh [CAN interface name] [bitrate]
#                For example, specify the interface name as my_can_interface and the bitrate as 1000000:
#                sudo bash ./can_config.sh my_can_interface 1000000
#             2. You can also specify the CAN name via a USB hardware address:
#                sudo bash ./can_config.sh [CAN interface name] [bitrate] [USB hardware address]
#                For example, specifying the interface name as my_can_interface, the bitrate as 1000000, and the USB hardware address as 1-3:1.0:
#                sudo bash ./can_config.sh my_can_interface 1000000 1-3:1.0
#                This means we assign the CAN device at USB address 1-3:1.0 the name my_can_interface with a bitrate of 1000000.
#          2. Multiple CAN modules
#             For multiple CAN modules, specify the USB_PORTS array in the script to set each CAN module’s interface name and bitrate.
#             No extra parameters are needed; just run the script:
#             sudo ./can_config.sh
#
# Notes:
#
#    Permissions:
#       The script must be run with sudo privileges, since renaming and configuring network interfaces require admin privileges.
#       Ensure you have sufficient permissions to run this script.
#
#    Script environment:
#       This script assumes it is running in a bash environment. Make sure your system uses bash instead of another shell (like sh).
#       You can verify this by checking the shebang line (#!/bin/bash).
#
#    USB port information:
#       Make sure the predefined USB port information (bus-info) matches what you actually see when running ethtool.
#       Use sudo ethtool -i can0, sudo ethtool -i can1, etc. to check each CAN interface’s bus-info.
#
#    Interface conflicts:
#       Ensure that the target interface names (like can_device_1, can_device_2) are unique and do not conflict with existing system interfaces.
#       If you need to modify the mapping between USB ports and interface names, adjust the USB_PORTS array accordingly.
#-------------------------------------------------------------------------------------------------#

# The predefined number of CAN modules
EXPECTED_CAN_COUNT=2

if [ "$EXPECTED_CAN_COUNT" -eq 1 ]; then
    # The default CAN name, can be set by the user via command-line arguments
    DEFAULT_CAN_NAME="${1:-can0}"

    # The default bitrate for a single CAN module, can be set by the user via command-line arguments
    DEFAULT_BITRATE="${2:-1000000}"

    # USB hardware address (optional parameter)
    USB_ADDRESS="${3}"
fi

# The predefined USB ports, target interface names, and their bitrates (used when multiple CAN modules are present)
if [ "$EXPECTED_CAN_COUNT" -ne 1 ]; then
    declare -A USB_PORTS 
    USB_PORTS["1-1:1.0"]="can_left:1000000"
    USB_PORTS["1-2:1.0"]="can_right:1000000"
fi

# Get the current number of CAN modules in the system
CURRENT_CAN_COUNT=$(ip link show type can | grep -c "link/can")

# Check if the current number of CAN modules in the system matches the expected number
if [ "$CURRENT_CAN_COUNT" -ne "$EXPECTED_CAN_COUNT" ]; then
    echo "Error: The detected number of CAN modules ($CURRENT_CAN_COUNT) does not match the expected number ($EXPECTED_CAN_COUNT)."
    exit 1
fi

# Load the gs_usb module
sudo modprobe gs_usb
if [ $? -ne 0 ]; then
    echo "Error: Unable to load the gs_usb module."
    exit 1
fi

# Determine if we only need to handle a single CAN module
if [ "$EXPECTED_CAN_COUNT" -eq 1 ]; then
    if [ -n "$USB_ADDRESS" ]; then
        echo "Detected USB hardware address parameter: $USB_ADDRESS"
        
        # Use ethtool to find the CAN interface corresponding to the USB hardware address
        INTERFACE_NAME=""
        for iface in $(ip -br link show type can | awk '{print $1}'); do
            BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
            if [ "$BUS_INFO" == "$USB_ADDRESS" ]; then
                INTERFACE_NAME="$iface"
                break
            fi
        done
        
        if [ -z "$INTERFACE_NAME" ]; then
            echo "Error: Could not find a CAN interface corresponding to the USB hardware address $USB_ADDRESS."
            exit 1
        else
            echo "Found interface $INTERFACE_NAME corresponding to USB hardware address $USB_ADDRESS"
        fi
    else
        # Get the unique CAN interface
        INTERFACE_NAME=$(ip -br link show type can | awk '{print $1}')
        
        # Check if we got the interface name
        if [ -z "$INTERFACE_NAME" ]; then
            echo "Error: Could not detect a CAN interface."
            exit 1
        fi

        echo "Only one CAN module expected, detected interface $INTERFACE_NAME"
    fi

    # Check if the current interface is already up
    IS_LINK_UP=$(ip link show "$INTERFACE_NAME" | grep -q "UP" && echo "yes" || echo "no")

    # Get the current bitrate of the interface
    CURRENT_BITRATE=$(ip -details link show "$INTERFACE_NAME" | grep -oP 'bitrate \K\d+')

    if [ "$IS_LINK_UP" == "yes" ] && [ "$CURRENT_BITRATE" -eq "$DEFAULT_BITRATE" ]; then
        echo "Interface $INTERFACE_NAME is already up with a bitrate of $DEFAULT_BITRATE"
        
        # Check if the interface name matches the default name
        if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
            echo "Renaming interface $INTERFACE_NAME to $DEFAULT_CAN_NAME"
            sudo ip link set "$INTERFACE_NAME" down
            sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
            sudo ip link set "$DEFAULT_CAN_NAME" up
            echo "Interface renamed to $DEFAULT_CAN_NAME and brought up again."
        else
            echo "Interface name is already $DEFAULT_CAN_NAME"
        fi
    else
        # If the interface is not up or the bitrate is different, set it
        if [ "$IS_LINK_UP" == "yes" ]; then
            echo "Interface $INTERFACE_NAME is up, but its bitrate is $CURRENT_BITRATE, which does not match $DEFAULT_BITRATE."
        else
            echo "Interface $INTERFACE_NAME is not up or its bitrate is not set."
        fi
        
        # Set interface bitrate and bring it up
        sudo ip link set "$INTERFACE_NAME" down
        sudo ip link set "$INTERFACE_NAME" type can bitrate $DEFAULT_BITRATE
        sudo ip link set "$INTERFACE_NAME" up
        echo "Interface $INTERFACE_NAME has been reconfigured to a bitrate of $DEFAULT_BITRATE and brought up."

        # Rename the interface to the default name
        if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
            echo "Renaming interface $INTERFACE_NAME to $DEFAULT_CAN_NAME"
            sudo ip link set "$INTERFACE_NAME" down
            sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
            sudo ip link set "$DEFAULT_CAN_NAME" up
            echo "Interface renamed to $DEFAULT_CAN_NAME and brought up again."
        fi
    fi
else
    # Handle multiple CAN modules

    # Check whether the number of USB ports and target interface names matches the expected CAN module count
    PREDEFINED_COUNT=${#USB_PORTS[@]}
    if [ "$EXPECTED_CAN_COUNT" -ne "$PREDEFINED_COUNT" ]; then
        echo "Error: The expected number of CAN modules ($EXPECTED_CAN_COUNT) does not match the number of predefined USB ports ($PREDEFINED_COUNT)."
        exit 1
    fi

    # Iterate over all CAN interfaces
    for iface in $(ip -br link show type can | awk '{print $1}'); do
        # Use ethtool to get bus-info
        BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
        
        if [ -z "$BUS_INFO" ];then
            echo "Error: Could not retrieve bus-info for interface $iface."
            continue
        fi
        
        echo "Interface $iface is plugged into USB port $BUS_INFO"

        # Check if the bus-info is in the predefined USB port list
        if [ -n "${USB_PORTS[$BUS_INFO]}" ];then
            IFS=':' read -r TARGET_NAME TARGET_BITRATE <<< "${USB_PORTS[$BUS_INFO]}"
            
            # Check if the interface is up
            IS_LINK_UP=$(ip link show "$iface" | grep -q "UP" && echo "yes" || echo "no")

            # Get the current interface bitrate
            CURRENT_BITRATE=$(ip -details link show "$iface" | grep -oP 'bitrate \K\d+')

            if [ "$IS_LINK_UP" == "yes" ] && [ "$CURRENT_BITRATE" -eq "$TARGET_BITRATE" ]; then
                echo "Interface $iface is already up with a bitrate of $TARGET_BITRATE"
                
                # Check if the interface name matches the target name
                if [ "$iface" != "$TARGET_NAME" ]; then
                    echo "Renaming interface $iface to $TARGET_NAME"
                    sudo ip link set "$iface" down
                    sudo ip link set "$iface" name "$TARGET_NAME"
                    sudo ip link set "$TARGET_NAME" up
                    echo "Interface renamed to $TARGET_NAME and brought up again."
                else
                    echo "Interface name is already $TARGET_NAME"
                fi
            else
                # If the interface is not up or the bitrate is different, configure it
                if [ "$IS_LINK_UP" == "yes" ]; then
                    echo "Interface $iface is up, but its bitrate is $CURRENT_BITRATE, which does not match $TARGET_BITRATE."
                else
                    echo "Interface $iface is not up or its bitrate is not set."
                fi
                
                # Set the interface bitrate and bring it up
                sudo ip link set "$iface" down
                sudo ip link set "$iface" type can bitrate $TARGET_BITRATE
                sudo ip link set "$iface" up
                echo "Interface $iface has been reconfigured to a bitrate of $TARGET_BITRATE and brought up."

                # Rename the interface to the target name
                if [ "$iface" != "$TARGET_NAME" ]; then
                    echo "Renaming interface $iface to $TARGET_NAME"
                    sudo ip link set "$iface" down
                    sudo ip link set "$iface" name "$TARGET_NAME"
                    sudo ip link set "$TARGET_NAME" up
                    echo "Interface renamed to $TARGET_NAME and brought up again."
                fi
            fi
        else
            echo "Error: Unknown USB port $BUS_INFO for interface $iface."
            exit 1
        fi
    done
fi

echo "All CAN interfaces have been successfully renamed and brought up."
