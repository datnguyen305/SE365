import os 
import shutil 
import random


class Preprocessor:
    def __init__(self, config):
        self.seed = config['seed']
        self.src = config['src_dir']
        self.tgt = config['tgt_dir']
        self.train = config['train']
        self.dev = config['dev']
        self.test = config['test']
        self.splits = {
            "train": self.train,
            "dev": self.dev,
            "test": self.test
        }
    def run(self):
        random.seed(self.seed)

        splits = {
            "train": self.train,
            "dev": self.dev,
            "test": self.test
        }
        classes = os.listdir(self.src)

        for split in splits:
            for cls in classes:
                os.makedirs(
                    os.path.join(self.tgt, split, cls), exist_ok=True
                )
        
        for cls in classes:
            cls_path = os.path.join(self.src, cls)
            files = os.listdir(cls_path)
            random.shuffle(files)

            n_total = len(files)
            n_train = int(n_total * splits["train"])
            n_dev = int(n_total * splits["dev"])

            train_files = files[:n_train]
            dev_files = files[n_train:n_train + n_dev]
            test_files = files[n_train + n_dev:]

            for f in train_files:
                shutil.copy(
                    os.path.join(cls_path, f),
                    os.path.join(self.tgt, "train", cls, f)
                )
            
            for f in dev_files:
                shutil.copy(
                    os.path.join(cls_path, f),
                    os.path.join(self.tgt, "dev", cls, f)
                )
            for f in test_files:
                shutil.copy(
                    os.path.join(cls_path, f),
                    os.path.join(self.tgt, "test", cls, f)
                )
        print(f"{cls}: train={len(train_files)}, dev={len(dev_files)}, test={len(test_files)}")