import os
import shutil
import tarfile

from six.moves.urllib.error import URLError, HTTPError
from six.moves.urllib.request import urlopen
from tqdm import tqdm


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
