from dora import Node
import time

def main():
    node = Node("iphone-listener")
    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "image":
                # print("Delay from camera:", (float(event["metadata"]["timestamp"]) - time.time()))
                print(event["value"].to_numpy().shape)

if __name__ == "__main__":
    main()
