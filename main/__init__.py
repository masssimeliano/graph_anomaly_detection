if __name__ == "__main__":
    from main.read_dataset import read
    from main.create_graph import create

    data = read()
    create(data)