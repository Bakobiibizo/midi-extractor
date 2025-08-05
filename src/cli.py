import json
from pathlib import Path
from argparse import ArgumentParser
from pydantic import BaseModel

from datasets.stem_indexer import get_dataset_index, filter_index
from datasets.slakh_stem_dataset import SlakhStemDataset
from datasets.visualize_sample import visualize_sample

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    
    return parser.parse_args()


def main():
    args = get_arguments()
    
    data_path = Path(args.dataset)
    
    output_path = get_dataset_index(data_path)
    
    filtered_path = filter_index(output_path, debug=True, threshold_db=-35.0)    
    
    dataset = SlakhStemDataset(filtered_path)
    
    for i in range(5):  
        sample = dataset[i]
        
        visualize_sample(sample, output_path=data_path / f"sample_{i}.png")
        
if __name__ == "__main__":
    main()
    