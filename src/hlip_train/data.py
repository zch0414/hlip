import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from open_clip_train.data import *


class StudyInfo(object):
    def __init__(self, data_root, uid, scans, reports):
        # missing files
        if uid == 'BRAIN_UM_8852F6EC' and 'VPCT_Stroke___PERFUSION__1_5__Hr35_27-DIAMOX_CHALLENGE_SCAN_1-Protocol_BloodWindow' in scans:
            scans.remove('VPCT_Stroke___PERFUSION__1_5__Hr35_27-DIAMOX_CHALLENGE_SCAN_1-Protocol_BloodWindow')
        if uid == 'BRAIN_UM_775A760A' and 'VPCT_Stroke__PERFUSION__1_5__H20f_8-DIAMOX_CHALLENGE_SCAN_1-Protocol_2_BoneWindow' in scans:
            scans.remove('VPCT_Stroke__PERFUSION__1_5__H20f_8-DIAMOX_CHALLENGE_SCAN_1-Protocol_2_BoneWindow')
        if uid == 'BRAIN_UM_1EDAA6D2' and 'VPCT_Stroke___PERFUSION__1_5__Hr35_4-DIAMOX_CHALLENGE_SCAN_1-Protocol_BrainWindow' in scans:
            scans.remove('VPCT_Stroke___PERFUSION__1_5__Hr35_4-DIAMOX_CHALLENGE_SCAN_1-Protocol_BrainWindow')
        if uid == 'BRAIN_UM_1EDAA6D2' and 'VPCT_Stroke___PERFUSION__1_5__Hr35_25-DIAMOX_CHALLENGE_SCAN_1-Protocol_BrainWindow' in scans:
            scans.remove('VPCT_Stroke___PERFUSION__1_5__Hr35_25-DIAMOX_CHALLENGE_SCAN_1-Protocol_BrainWindow')
        if uid == 'BRAIN_UM_1F9E66BC' and 'VPCT_Stroke__PERFUSION__1_5__H20f_28-DIAMOX_CHALLENGE_SCAN_1-Protocol_2_BloodWindow' in scans:
            scans.remove('VPCT_Stroke__PERFUSION__1_5__H20f_28-DIAMOX_CHALLENGE_SCAN_1-Protocol_2_BloodWindow')
        if uid == 'BRAIN_UM_0D09E9AB' and 'VPCT_Stroke_PERFUSION__1_5__Hr35_2nd_30-DIAMOX_CHALLENGE_SCAN_1-Protocol_BloodWindow' in scans:
            scans.remove('VPCT_Stroke_PERFUSION__1_5__Hr35_2nd_30-DIAMOX_CHALLENGE_SCAN_1-Protocol_BloodWindow')
        if uid == 'BRAIN_UM_32C15A8B' and 'VPCT_Stroke__DynMulti4D__1_5__H20f_3-BRAIN_PERFUSION-Protocol_BloodWindow' in scans:
            scans.remove('VPCT_Stroke__DynMulti4D__1_5__H20f_3-BRAIN_PERFUSION-Protocol_BloodWindow')
        if uid == 'BRAIN_UM_3A37E8E4' and 'VPCT_Stroke__DynMulti4D__1_5__H20f_23-BRAIN_PERFUSION-Protocol_BrainWindow' in scans:
            scans.remove('VPCT_Stroke__DynMulti4D__1_5__H20f_23-BRAIN_PERFUSION-Protocol_BrainWindow')

        self.scans = np.array([os.path.join(data_root, uid, scan, 'img.pt') for scan in scans])
        self.report = np.array(reports)
    
    def get_sentence(self, idx, shuffle):
        if shuffle:
            return 'This study shows: ' + ' '.join(np.random.permutation(self.report).tolist()[idx: idx + 1])
        else:
            return 'This study shows: ' + ' '.join(self.report.tolist()[idx: idx + 1])
        
    def get_report(self, shuffle):
        if shuffle:
            return 'This study shows: ' + ' '.join(np.random.permutation(self.report).tolist())
        else:
            return 'This study shows: ' + ' '.join(self.report.tolist())

    def get_scan(self, shuffle):
        if shuffle:
            return np.random.permutation(self.scans).tolist()
        else:
            return self.scans.tolist()


class StudyDataset(Dataset):
    def __init__(
        self, 
        data_filelist, # a list of data root
        scan_filedict, # a dictionary {data_root: scan_filelist}
        report_filedict, # a dictionary {data_root: report_filelist}
        uid_filedict=None, # a dictionary {data_root: uid_filelist}
        num_scans=None,
        tokenizer=None,
        is_train=False,
    ):
        self.studies = []
        for data_file in data_filelist:
            scan_filelist = scan_filedict[data_file]
            report_filelist = report_filedict[data_file]
            uid2scans, uid2reports = {}, {}

            for scan_json_file in scan_filelist:
                with open(scan_json_file, 'r') as file:
                    data = json.load(file)
                uid2scans.update(data)
            for report_json_file in report_filelist:
                with open(report_json_file, 'r') as file:
                    data = json.load(file)
                uid2reports.update(data)
            
            if uid_filedict:
                uids = []
                uid_filelist = uid_filedict[data_file]
                for uid_file in uid_filelist:
                    data = pd.read_csv(uid_file)["uid"].tolist()
                    uids.extend(data)
            else:
                uids = [u for u in uid2reports.keys() if u in uid2scans] # NOTE: DON NOT use list(uid2scans.keys() & uid2reports.keys()), which causes problem for DDP
            
            for uid in uids:
                self.studies.append(StudyInfo(data_file, uid, uid2scans[uid], uid2reports[uid]))

        # debug
        # self.studies = self.studies[: 1536]

        self.num_scans = num_scans
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
    
    def __len__(self):
        return len(self.studies)
    
    def __getitem__(self, idx):
        study = self.studies[idx] # get study
        
        sentence = self.tokenizer([str(study.get_sentence(idx=0, shuffle=self.is_train))])[0] # get sentence
        report = self.tokenizer([str(study.get_report(shuffle=self.is_train))])[0] # get report
        
        scans = study.get_scan(shuffle=self.is_train) # get scans
        repeats = -(-self.num_scans // len(scans)) if self.is_train else 1
        scans *= repeats
        scans = scans[:self.num_scans] if self.is_train else scans
        
        image = [] # load-in scans
        for scan in scans:
            img = torch.load(scan, weights_only=True)
            img = img.float() / 255.0
            img = self.normalizer(img[None, ...])[0]
            image.append(img[None, ...])

        # NOTE: convert image to torch.float16 by default
        return {'image': torch.stack(image, dim=0).to(dtype=torch.float16), 'sentence': sentence, 'report': report}


def get_dataset(args, tokenizer, is_train):
    dataset = StudyDataset(
        data_filelist=args.train_data_filelist if is_train else args.valid_data_filelist,
        scan_filedict=args.train_scan_filedict if is_train else args.valid_scan_filedict,
        report_filedict=args.train_report_filedict if is_train else args.valid_report_filedict,
        uid_filedict=args.train_uid_filedict if is_train else None,
        num_scans=args.num_scans, 
        tokenizer=tokenizer, 
        is_train=is_train,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else 1, # avoid CPU memory issue
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
        data['train'] = get_dataset(args, tokenizer=tokenizer, is_train=True)
    if args.valid_data:
        data['valid'] = get_dataset(args, tokenizer=tokenizer, is_train=False)
    if args.mri: # internal zero-shot evaluation for mri
        from hlip_test.zeroshot_mri import get_data as get_mri
        data['mri'] = get_mri(
            data_root=args.mri['data_root'],
            input_file=args.mri['input_file'],
            workers=args.workers,
            distributed=args.distributed
        )
    if args.ct: # internal zero-shot evaluation for ct
        from hlip_test.zeroshot_ct import get_data as get_ct
        data['ct'] = get_ct(
            data_root=args.ct['data_root'],
            input_file=args.ct['input_file'],
            workers=args.workers,
            distributed=args.distributed
        )
    return data