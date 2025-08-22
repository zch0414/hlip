import os
import zipfile
import logging

from tqdm import tqdm
from pathlib import Path


def unzip_file(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_items = zip_ref.namelist()
        for item in zip_items:
            try:
                zip_ref.extract(item, extract_to)
                logging.info(f"Extracted: {item}")
            except zipfile.BadZipFile:
                logging.warning(f"Bad zip file: {item}")
            except zipfile.LargeZipFile:
                logging.warning(f"File too large: {item}")
            except Exception as e:
                logging.warning(f"Failed to extract {item}: {e}")


if __name__ == "__main__":
    logging.basicConfig(filename='unzip.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    root_path = '/path/to/brats23'
    files = sorted([p for p in Path(root_path).rglob("*.zip")])
    for file in files:
        logging.info(f"unzip {str(file).split('/')[-2]}: {str(file).split('/')[-1]} ...")
        extract_to = '/'.join(str(file).split('.')[0].split('/')[:-1])
        unzip_file(file, extract_to)
    logging.info("Done!")