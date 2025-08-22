import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from torchvision.transforms import Normalize
from open_clip_train.data import *

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class StudyInfo(object):
    def __init__(self, root, key, value):
        self.scans = np.array([os.path.join(root, key, scans, 'img.pt') for scans in value['scans']])
        self.report = np.array(value['report'])
    
    def get_report(self, shuffle):
        if shuffle:
            return 'This MRI study shows: ' + ' '.join(np.random.permutation(self.report).tolist())
        else:
            return 'This MRI study shows: ' + ' '.join(self.report.tolist())

    def get_scans(self, shuffle):
        if shuffle:
            return np.random.permutation(self.scans).tolist()
        else:
            return self.scans.tolist()


class StudyDataset(Dataset):
    def __init__(
        self, 
        json_root, data_root, input_filename, 
        transform=None, 
        tokenizer=None,
        is_train=True,
        num_scans=None,
    ):
        with open(os.path.join(json_root, input_filename + '.json'), 'r') as file:
            studies = json.load(file)
        self.studies = [StudyInfo(root=os.path.join(data_root, input_filename), key=key, value=value) for key, value in studies.items()]

        self.is_train = is_train
        self.num_scans = num_scans
        
        self.transform = transform
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.studies)
    
    def __getitem__(self, idx):
        study = self.studies[idx]

        # get report
        report = self.tokenizer([str(study.get_report(shuffle=self.is_train))])[0]

        # get scans
        scans = study.get_scans(shuffle=self.is_train)
        repeats = -(-self.num_scans // len(scans))
        scans *= repeats
        scans = scans[:self.num_scans] if self.is_train else scans
        
        # load-in scans
        imgs = []
        for s in scans:
            img = torch.load(s, weights_only=True)
            img = img[None, ...].float() / 255.0 # [1, d, h, w]

            if self.transform:
                img = self.transform(img)
                img = torch.as_tensor(img).float()

            normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
            img = normalizer(img)
            imgs.append(img)

        return torch.stack(imgs, dim=0), report
    

def get_dataset(args, preprocess_fn, is_train, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = StudyDataset(
        args.json_root, args.data_root, input_filename,
        preprocess_fn,
        tokenizer,
        is_train=is_train,
        num_scans=args.num_scans,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else 1,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_data(args, tokenizer=None):
    data = {}
    if args.train_data:
        data["train"] = get_dataset(args, None, is_train=True, tokenizer=tokenizer)
    if args.val_data:
        data["val"] = get_dataset(args, None, is_train=False, tokenizer=tokenizer)
    return data