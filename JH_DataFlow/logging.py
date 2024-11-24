import logging
import sys

from datetime import datetime
from pytz import timezone

from pathlib import Path
from typing import Dict, Optional



def timetz(*args):
    tz = timezone('Asia/Shanghai')
    return datetime.now(tz).timetuple()


def set_logging(filename: Optional[str] = None):
    '''
    Args:
        filename: str default None
    '''
    if filename is None:
        filename = f'/tmp/log_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
        print(f'filename: {filename}')
        
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    params = {}
    if filename is not None:
        params.update({'filename': filename})
    params.update({
        'level': logging.INFO,
        'filemode': 'a',
        'format': '[%(asctime)s] [%(levelname)s]: %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    })
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(**params)
    logging.Formatter.converter = timetz
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    return


def log_params(**kwargs: Dict) -> None:
    for k, v in kwargs.items():
        logging.info(f'{k}: {v}')
    return
    