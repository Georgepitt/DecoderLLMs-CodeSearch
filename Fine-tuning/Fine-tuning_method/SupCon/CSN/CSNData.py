import json
import random
import os
import glob

from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

datasets_list = [
    "go",
    "java",
    "javascript",
    "php",
    "python",
    "ruby",
]

CSN_EMBEDDING_PROMPTS = {
    "go": "Given a code query, retrieve the code snippet that answers it",
    "java": "Given a code query, retrieve the code snippet that answers it",
    "javascript": "Given a code query, retrieve the code snippet that answers it",
    "php": "Given a code query, retrieve the code snippet that answers it",
    "python": "Given a code query, retrieve the code snippet that answers it",
    "ruby": "Given a code query, retrieve the code snippet that answers it",
}


class CSNData(Dataset):
    def __init__(
        self,
        dataset_name: str = "CSN",
        split: str = "validation",
        file_path: str = "CodeSearchNet/resources",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        logger.info(f"Loading CSN data from {file_path}...")
        # file path is actually a directory

        data_map = {}
        all_samples = []
        id_ = 0
        # Data list path traversal
        for dataset in datasets_list:
            logger.info(f"Loading {dataset} data")
            if dataset not in data_map:
                data_map[dataset] = []
            basic_path = os.path.join(file_path, dataset , 'train', '*.jsonl')
            file_paths = glob.glob(basic_path)
            logger.info(f"Found {len(file_paths)} files for dataset {dataset}")
            logger.info(f"File paths for dataset {dataset}: {file_paths}")

            for data_path in file_paths:
                logger.info(f"Loading dataset {data_path}...")

                with open(data_path, 'r', encoding='utf-8') as f:
                    content = f.read() 
                    dataset_samples = json.loads(content)  

                logger.info(f"Found {len(dataset_samples)} samples in {data_path}")



                # Process each sample with query=prompt+ separator + query in sample
                for sample in dataset_samples:
                    query = sample["query"]
                    pos = sample["positive"]
                    neg = sample["negative"]
                    # Create a DataSample instance and add it to the all_samples list
                    data_sample = DataSample(
                        id_=id_,
                        query=query,
                        positive=pos,
                        negative=neg,
                        task_name=dataset,
                    )
                    all_samples.append(data_sample)
                    data_map[dataset].append(id_)
                    id_ += 1


        # combine split1 and split2
        new_data_map = {}
        for dataset in data_map:
            new_dataset = dataset
            if new_dataset not in new_data_map:
                new_data_map[new_dataset] = []
            new_data_map[new_dataset] += data_map[dataset]
        data_map = new_data_map

        if self.shuffle_individual_datasets:
            random.seed(42)  
            for task, samples in data_map.items():
                random.shuffle(samples)

        datasets = list(data_map.keys())

        logger.info(
            f"Batching Echo data properly for effective batch size of {self.effective_batch_size}..."
        )
        # Batch the data
        all_batches = []
        for dataset in datasets:
            dataset_samples = data_map[dataset]
            for i in range(0, len(dataset_samples), self.effective_batch_size):
                batch = dataset_samples[i : i + self.effective_batch_size]
                if len(batch) == self.effective_batch_size:
                    all_batches.append(batch)
                else:
                    logger.info(f"Skip 1 batch for dataset {dataset}.")
        random.shuffle(all_batches)

        final_idx_order = []
        for batch in all_batches:
            for idx in batch:
                final_idx_order.append(idx)
        # Store all samples in self.data
        self.data = [all_samples[idx] for idx in final_idx_order]
        logger.info(f"Loaded {len(self.data)} samples.")

    # Return the corresponding data sample according to the index
    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive, sample.negative], label=1.0
            )
        elif self.split == "validation":
            assert False, "CSNData does not have a validation split."
