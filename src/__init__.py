if __name__ == "__main__":
    from src.utils.read_and_form_dataset import form
    from src.utils.draw_graph import draw
    from src.models import baseline_model as baseline
    from src.models import structure_and_feature_model as structure_and_feature

    import time

    print("Reading")
    labels, graph = form()

    print("Drawing")
    nx_graph = draw(graph)

    print("Begin training")
    start = time.time()
    baseline.train(nx_graph, labels)
    end = time.time()
    print("End training")
    print(f"Execution time: {end - start:.4f} sec")
    print("")

    print("Begin training")
    start = time.time()
    structure_and_feature.train(nx_graph, labels)
    end = time.time()
    print("End training")
    print(f"Execution time: {end - start:.4f} sec")
    print("")