from datasets import load_dataset

def load_hf_dataset(name: str, split: str = None):
    """
    Loads a dataset from Hugging Face Hub.
    
    Args:
        name (str): Dataset name (e.g., 'imdb')
        split (str): Optional split ('train', 'test')
    
    Returns:
        Dataset or DatasetDict
    """
    dataset = load_dataset(name, split=split)
    return dataset
