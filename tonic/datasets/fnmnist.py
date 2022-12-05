import os
from typing import Callable, Optional

import numpy as np

from tonic.dataset import Dataset
from tonic.io import read_fixation_mnist_file


class FNMNIST(Dataset):
    """`FN-MNIST <https://data.mendeley.com/datasets/559n9zbb7c/2>`_

    Events have (xytp) ordering.
    ::

        @article{,
          title={},
          author={},
          journal={},
          volume={},
          pages={},
          year={},
          publisher={}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        duration (float): Maximum duration to consider for each sample in microseconds.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.
    """

    base_url = "https://data.mendeley.com/datasets/559n9zbb7c/2/files/"
    train_url = base_url + "47229907-4b2e-4e76-b2cf-dc38f01419cb/file_downloaded"
    train_filename = "train.zip"
    train_md5 = "5e24ef5f41d48b53642ca9be5a70ca75"
    train_folder = "Train"
    test_url = base_url + "88f0de5b-2d86-41cd-a8f2-1fb42cddc06c/file_downloaded"
    test_filename = "test.zip"
    test_md5 = "70ee55f776304fa5777eea893fa3a7f6"
    test_folder = "Test"

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    sensor_size = (34, 34, 2)
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        train: bool = True,
        duration: float = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.train = train
        self.duration = duration

        if train:
            self.filename = self.train_filename
            self.url = self.train_url
            self.file_md5 = self.train_md5
            self.folder_name = self.train_folder
        else:
            self.filename = self.test_filename
            self.url = self.test_url
            self.file_md5 = self.test_md5
            self.folder_name = self.test_folder

        if not self._check_exists():
            self.download()

        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            files.sort()
            for file in files:
                if file.endswith("bin"):
                    self.data.append(path + "/" + file)
                    label_number = int(path[-1])
                    self.targets.append(label_number)

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = read_fixation_mnist_file(self.data[index], dtype=self.dtype)
        if self.duration:
            events = events[events["t"] < self.duration]
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)
        return events, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_exists(self) -> bool:
        return (
            self._is_file_present()
            and self._folder_contains_at_least_n_files_of_type(10000, ".bin")
        )
