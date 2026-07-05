"""
NVIDIA DALI data pipeline for training (GPU decode + normalize).
Requires Python 3.8–3.12 (DALI/dm-tree do not support 3.14 yet).
Install: uv pip install nvidia-dali-cuda130  (CUDA 13; or cuda120 / cuda110).
Check CUDA: nvidia-smi
"""
import os
import math

try:
    from nvidia.dali import pipeline_def, Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    LastBatchPolicy = None


def get_class_nums_and_size(file_list):
    """Read annotation file (path label per line) and return (num_classes, num_samples)."""
    paths = []
    labels = []
    with open(file_list) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                paths.append(parts[0])
                labels.append(int(parts[1]))
    num_classes = len(set(labels))
    return num_classes, len(paths)


@pipeline_def(batch_size=256, num_threads=4, device_id=0)
def train_pipeline(file_root, file_list, crop_size=112):
    """DALI pipeline: read file list, decode on GPU, random flip, normalize to [-1,1]."""
    jpegs, labels = fn.readers.file(
        file_root=file_root,
        file_list=file_list,
        random_shuffle=True,
        name="Reader",
    )
    images = fn.decoders.image(jpegs, device="mixed")
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        crop=(crop_size, crop_size),
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        mirror=fn.random.coin_flip(),
        output_layout=types.NCHW,
    )
    return images, labels


class DALITrainIteratorWrapper:
    """Wraps DALIGenericIterator to yield (img, label) like PyTorch DataLoader.
    DALI returns list of dicts per GPU; labels may be (N,1) -> squeeze to (N,).
    """

    def __init__(self, dali_iter):
        self.dali_iter = dali_iter

    def __iter__(self):
        for batch in self.dali_iter:
            data = batch[0]["data"]
            label = batch[0]["label"]
            if label.dim() > 1:
                label = label.squeeze(-1)
            label = label.long()
            yield data, label


def create_dali_train_loader(
    file_root,
    file_list,
    batch_size,
    device_id=0,
    num_threads=4,
    crop_size=112,
    last_batch_policy=None,
):
    """
    Create DALI pipeline and iterator for training.
    Returns (iterator_wrapper, num_classes, num_samples).
    """
    if not DALI_AVAILABLE:
        raise ImportError(
            "DALI not installed. Install with: uv pip install nvidia-dali-cuda130  (or cuda120 / cuda110)"
        )
    file_root = os.path.abspath(file_root)
    file_list = os.path.abspath(file_list)
    num_classes, num_samples = get_class_nums_and_size(file_list)
    pipe = train_pipeline(
        file_root=file_root,
        file_list=file_list,
        crop_size=crop_size,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
    )
    pipe.build()
    if last_batch_policy is None:
        last_batch_policy = LastBatchPolicy.PARTIAL
    dali_iter = DALIGenericIterator(
        [pipe],
        ["data", "label"],
        reader_name="Reader",
        last_batch_policy=last_batch_policy,
        auto_reset=True,
    )
    wrapper = DALITrainIteratorWrapper(dali_iter)
    return wrapper, num_classes, num_samples
