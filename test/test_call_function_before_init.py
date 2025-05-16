from robot.rpc import RPCClient

def main():
    cone_e = RPCClient('localhost', 5000)

    print(cone_e.get_lift_position())
    input("Press Enter to exit")

if __name__ == "__main__":
    main()
