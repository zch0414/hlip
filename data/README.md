## CT-RATE

Downloading the CT-RATE dataset with:
```bash
python download_valid.py
python download_train.py
```
Downloading the training split may take ~2 days. Once complete, process the dataset with:
```bash
python process.py --num-cpus 8 --data 'valid' --root-dir '/download/ct_rate/dataset/' --save-dir '/data/ct_rate/'
python process.py --num-cpus 8 --data 'train' --root-dir '/download/ct_rate/dataset/' --save-dir '/data/ct_rate/'
```
Based on our analysis, data type and spacing should not be critical concerns. One may consider the following commands to reduce the dataset size:
<code>--save-astype</code> and <code>--spacing</code>.

All necessary files have already been provided, some of which are provided by [fVLM](https://github.com/alibaba-damo-academy/fvlm). 
There is no need to run the other code; it is included solely as a reference to illustrate how the files were generated.

**reference:**
```bib
@misc{hamamci2024foundation,
  title={Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography}, 
  author={Ibrahim Ethem Hamamci and Sezgin Er and Furkan Almas and Ayse Gulnihan Simsek and Sevval Nil Esirgun and Irem Dogan and Muhammed Furkan Dasdelen and Omer Faruk Durugol and Bastian Wittmann and Tamaz Amiranashvili and Enis Simsar and Mehmet Simsar and Emine Bensu Erdemir and Abdullah Alanbay and Anjany Sekuboyina and Berkan Lafci and Christian Bluethgen and Mehmet Kemal Ozdemir and Bjoern Menze},
  year={2024},
  eprint={2403.17834},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2403.17834}, 
}
```

## Rad-ChestCT

Downloading the Rad-ChestCT dataset with:
```bash
python download.py
```
Once complete, process the dataset with:
```bash
python process.py --num-cpus 8 --root-dir '/download/rad_chestct/' --save-dir '/data/rad_chestct/'
```
All necessary files have already been provided. There is no need to run the other code; it is included solely as a reference to illustrate how the files were generated.

**reference:**
```bib
@article{draelos2021machine,
  title={Machine-learning-based multiple abnormality prediction with large-scale chest computed tomography volumes},
  author={Draelos, Rachel Lea and Dov, David and Mazurowski, Maciej A and Lo, Joseph Y and Henao, Ricardo and Rubin, Geoffrey D and Carin, Lawrence},
  journal={Medical image analysis},
  volume={67},
  pages={101857},
  year={2021},
  publisher={Elsevier}
}
```

## Pub-Brain-5

We provide the <code>uid</code> of [Open-BHB](https://baobablab.github.io/bhb/dataset), [Stroke](https://www.icpsr.umich.edu/web/ICPSR/studies/38464), [BraTS23](https://www.synapse.org/Synapse:syn51156910/wiki/627000), [NYUMets](https://nyumets.org/docs/brainapi/), and [UCSFMets](https://imagingdatasets.ucsf.edu/dataset/1). All of these datasets are publicly available. Due to licensing restrictions, Pub-Brain-5 can only be reconstructed using the provided <code>uid</code>. Reference code and process logs are included for convenience.

**reference:**
```bib
@article{dufumier2022openbhb,
  title={Openbhb: a large-scale multi-site brain mri data-set for age prediction and debiasing},
  author={Dufumier, Benoit and Grigis, Antoine and Victor, Julie and Ambroise, Corentin and Frouin, Vincent and Duchesnay, Edouard},
  journal={NeuroImage},
  volume={263},
  pages={119637},
  year={2022},
  publisher={Elsevier}
}

@article{liu2023large,
  title={A large public dataset of annotated clinical MRIs and metadata of patients with acute stroke},
  author={Liu, Chin-Fu and Leigh, Richard and Johnson, Brenda and Urrutia, Victor and Hsu, Johnny and Xu, Xin and Li, Xin and Mori, Susumu and Hillis, Argye E and Faria, Andreia V},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={548},
  year={2023},
  publisher={Nature Publishing Group UK London}
}

@article{baid2021rsna,
  title={The rsna-asnr-miccai brats 2021 benchmark on brain tumor segmentation and radiogenomic classification},
  author={Baid, Ujjwal and Ghodasara, Satyam and Mohan, Suyash and Bilello, Michel and Calabrese, Evan and Colak, Errol and Farahani, Keyvan and Kalpathy-Cramer, Jayashree and Kitamura, Felipe C and Pati, Sarthak and others},
  journal={arXiv preprint arXiv:2107.02314},
  year={2021}
}

@misc{labella2023asnrmiccai,
      title={The ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2023: Intracranial Meningioma}, 
      author={Dominic LaBella and Maruf Adewole and Michelle Alonso-Basanta and Talissa Altes and Syed Muhammad Anwar and Ujjwal Baid and Timothy Bergquist and Radhika Bhalerao and Sully Chen and Verena Chung and Gian-Marco Conte and Farouk Dako and James Eddy and Ivan Ezhov and Devon Godfrey and Fathi Hilal and Ariana Familiar and Keyvan Farahani and Juan Eugenio Iglesias and Zhifan Jiang and Elaine Johanson and Anahita Fathi Kazerooni and Collin Kent and John Kirkpatrick and Florian Kofler and Koen Van Leemput and Hongwei Bran Li and Xinyang Liu and Aria Mahtabfar and Shan McBurney-Lin and Ryan McLean and Zeke Meier and Ahmed W Moawad and John Mongan and Pierre Nedelec and Maxence Pajot and Marie Piraud and Arif Rashid and Zachary Reitman and Russell Takeshi Shinohara and Yury Velichko and Chunhao Wang and Pranav Warman and Walter Wiggins and Mariam Aboian and Jake Albrecht and Udunna Anazodo and Spyridon Bakas and Adam Flanders and Anastasia Janas and Goldey Khanna and Marius George Linguraru and Bjoern Menze and Ayman Nada and Andreas M Rauschecker and Jeff Rudie and Nourel Hoda Tahon and Javier Villanueva-Meyer and Benedikt Wiestler and Evan Calabrese},
      year={2023},
      eprint={2305.07642},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{moawad2024brain,
  title={The Brain Tumor Segmentation-Metastases (BraTS-METS) Challenge 2023: Brain Metastasis Segmentation on Pre-treatment MRI},
  author={Moawad, Ahmed W and Janas, Anastasia and Baid, Ujjwal and Ramakrishnan, Divya and Saluja, Rachit and Ashraf, Nader and Maleki, Nazanin and Jekel, Leon and Yordanov, Nikolay and Fehringer, Pascal and others},
  journal={ArXiv},
  pages={arXiv--2306},
  year={2024}
}

@article{kazerooni2024brain,
  title={The brain tumor segmentation (BraTS) challenge 2023: focus on pediatrics (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs)},
  author={Kazerooni, Anahita Fathi and Khalili, Nastaran and Liu, Xinyang and Haldar, Debanjan and Jiang, Zhifan and Anwar, Syed Muhammed and Albrecht, Jake and Adewole, Maruf and Anazodo, Udunna and Anderson, Hannah and others},
  journal={ArXiv},
  pages={arXiv--2305},
  year={2024}
}

@article{link2024longitudinal,
  title={Longitudinal deep neural networks for assessing metastatic brain cancer on a large open benchmark},
  author={Link, Katherine E and Schnurman, Zane and Liu, Chris and Kwon, Young Joon and Jiang, Lavender Yao and Nasir-Moin, Mustafa and Neifert, Sean and Alzate, Juan Diego and Bernstein, Kenneth and Qu, Tanxia and others},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={8170},
  year={2024},
  publisher={Nature Publishing Group UK London}
}

@article{rudie2024university,
  title={The University of California San Francisco Brain Metastases Stereotactic Radiosurgery (UCSF-BMSR) MRI Dataset},
  author={Rudie, Jeffrey D and Saluja, Rachit and Weiss, David A and Nedelec, Pierre and Calabrese, Evan and Colby, John B and Laguna, Benjamin and Mongan, John and Braunstein, Steve and Hess, Christopher P and others},
  journal={Radiology: Artificial Intelligence},
  volume={6},
  number={2},
  pages={e230126},
  year={2024},
  publisher={Radiological Society of North America}
}
```

