from robot.rpc import RPCClient

def main():
    cone_e = RPCClient('localhost', 5000)

    input("Press Enter to initialize ConeE")
    cone_e.init()
    input("Press Enter to exit")

if __name__ == "__main__":
    main()