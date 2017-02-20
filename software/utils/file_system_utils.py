import os
import shutil
import tarfile
import zipfile

from six.moves.urllib.error import URLError, HTTPError
from six.moves.urllib.request import urlopen
from tqdm import tqdm


#########################################################
# DOWNLOADS
#########################################################
def urlretrieve(url, file_path, progbar=None, data=None):
    """ From Keras
    (https://github.com/fchollet/keras/blob/master/keras/utils/data_utils.py)

    Args:
        url (str): The URL to retrieve.
        file_path (str): The path to store the retrieved data.
        progbar (tqdm object): A tqdm progress bar object.
        data: `data` argument passed to `urlopen`.
    """

    def chunk_read(response, chunk_size=8192, progbar=None):
        total_size = int(response.info().get('Content-Length').strip())
        progbar.total = total_size
        while 1:
            chunk = response.read(chunk_size)
            if not chunk:
                if progbar:
                    progbar.close()
                break
            if progbar:
                progbar.update(chunk_size)
            yield chunk

    response = urlopen(url, data)
    os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'wb') as fd:
        for chunk in chunk_read(response, progbar=progbar):
            fd.write(chunk)


def download(origin, file_path):
    """
    Args:
        origin (str): The origin of the file to download (i.e URL)
        file_path (str): The path where to download.
    """
    if os.path.exists(file_path):
        print('{} already exists. Not downloading again'.format(file_path))
        return
    print('Downloading data from {}'.format(origin))
    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
        try:
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=origin) as t:
                urlretrieve(origin, file_path, t)
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.msg))
    except (Exception, KeyboardInterrupt):
        if os.path.exists(file_path):
            os.remove(file_path)
        raise


#########################################################
# ARCHIVES
#########################################################
def get_extraction_root_name(archive_file_path):
    if tarfile.is_tarfile(archive_file_path):
        with tarfile.open(archive_file_path, 'r') as tfile:
            return tfile.getnames()[0]
    elif zipfile.is_zipfile(archive_file_path):
        with zipfile.ZipFile(archive_file_path, 'r') as zfile:
            return zfile.namelist()[0]
    else:
        msg = 'Unrecognized archive file type for {}'
        raise ValueError(msg.format(archive_file_path))


def untar(file_path, destination_folder, untar_path):
    if os.path.exists(untar_path):
        print('{} already exists. Not untaring.'.format(untar_path))
        return
    print('Untaring {}...'.format(file_path))
    with tarfile.open(file_path, 'r') as tfile:
        try:
            tfile.extractall(path=destination_folder)
        except (Exception, KeyboardInterrupt):
            if os.path.exists(untar_path):
                if os.path.isfile(untar_path):
                    os.remove(untar_path)
                else:
                    shutil.rmtree(untar_path)
            raise


def extract_all(archive_file_path, destination_folder=None):
    if destination_folder is None:
        destination_folder = os.path.dirname(archive_file_path)
    extract_name = get_extraction_root_name(archive_file_path)
    extract_path = os.path.join(destination_folder, extract_name)
    if os.path.exists(extract_path):
        print('{} already exists. Not extracting.'.format(extract_path))
    else:
        print('Extracting {}...'.format(archive_file_path))
        if tarfile.is_tarfile(archive_file_path):
            archive_file = tarfile.open(archive_file_path, 'r')
        elif zipfile.is_zipfile(archive_file_path):
            archive_file = zipfile.ZipFile(archive_file_path, 'r')
        else:
            msg = 'Unrecognized archive file type for {}'
            raise ValueError(msg.format(archive_file_path))
        try:
            archive_file.extractall(path=destination_folder)
        except (Exception, KeyboardInterrupt):
            if os.path.exists(extract_path):
                if os.path.isfile(extract_path):
                    os.remove(extract_path)
                else:
                    shutil.rmtree(extract_path)
            raise
        finally:
            archive_file.close()
    return extract_path
