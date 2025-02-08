from dora import Node

def main():
    node = Node("iphone-listener")
    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "image":
                print(event["value"].shape)
                print(event["metadata"]["timestamp"])
            if event["id"] == "depth":
                print(event["value"].shape)
                print(event["metadata"]["timestamp"])

if __name__ == "__main__":
    main()
