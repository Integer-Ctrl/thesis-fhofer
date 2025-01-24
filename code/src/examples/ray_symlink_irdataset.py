import os
import ray
import ir_datasets

ray.init()


@ray.remote
def symlink():
    os.symlink('/mnt/ceph/storage/data-tmp/current/ho62zoq/.ir_datasets/disks45/corpus/', '/home/ray/.ir_datasets/disks45/corpus')

    dataset = ir_datasets.load('disks45/nocr/trec-robust-2004')
    doc_store = dataset.docs_store()

    print(doc_store.get('FBIS3-15092'))


if __name__ == '__main__':

    NUM_WORKERS = 1

    ray.get([symlink.remote() for _ in range(NUM_WORKERS)])
