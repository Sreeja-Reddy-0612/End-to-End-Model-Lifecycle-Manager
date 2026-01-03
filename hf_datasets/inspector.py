def inspect_dataset(dataset):
    """
    Prints schema and basic statistics of the dataset.
    """
    print("🔹 Dataset Type:", type(dataset))
    print("🔹 Features:", dataset.features)

    # Show one example
    print("\n🔹 Sample record:")
    print(dataset[0])

    # Label distribution (if labels exist)
    if "label" in dataset.features:
        labels = [example["label"] for example in dataset]
        label_counts = {}
        for l in labels:
            label_counts[l] = label_counts.get(l, 0) + 1

        print("\n🔹 Label distribution:")
        for k, v in label_counts.items():
            print(f"Label {k}: {v}")
