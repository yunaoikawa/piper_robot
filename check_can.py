from phoenix6 import hardware

def main():
    print("=== Checking CANbus can0 ===")

    fx = hardware.TalonFX(1, "can0")   # Use can0 instead of Drivetrain
    supply = fx.get_supply_voltage().value

    print(f"TalonFX 1 supply voltage: {supply:.2f} V")

if __name__ == "__main__":
    main()
