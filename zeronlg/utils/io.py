import os
import json
import torch.distributed as dist
from typing import List
from .distributed import (
    is_dist_avail_and_initialized, 
    is_main_process, 
    get_rank, 
    get_world_size,
)
from sentence_transformers.util import snapshot_download
from .. import __LIBRARY_NAME__, __version__


def read_json(rpath: str):
    result = []
    with open(rpath, 'rt') as f:
        for line in f:
            result.append(json.loads(line.strip()))

    return result


def write_json(result: list, wpath: str):
    with open(wpath, 'wt') as f:
        for res in result:
            f.write(json.dumps(res) + '\n')


def collect_result(result, filename, local_wdir, save_result=False, remove_duplicate='', do_not_collect=False):
    assert isinstance(result, list)
    write_json(result, os.path.join(local_wdir,'%s_rank%d.json' % (filename, get_rank())))
    if is_dist_avail_and_initialized():
        dist.barrier()

    if do_not_collect:
        return None

    result = []
    final_result_file = ''
    if is_main_process():
        # combine results from all processes
        for rank in range(get_world_size()):
            result += read_json(os.path.join(local_wdir, '%s_rank%d.json' % (filename, rank)))

        if remove_duplicate:  # for evaluating captioning tasks
            result_new = []
            id_list = set()
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.add(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        if save_result:
            final_result_file = os.path.join(local_wdir, '%s.json' % filename)
            json.dump(result, open(final_result_file, 'w'), indent=4)
            print('result file saved to %s' % final_result_file)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return final_result_file if save_result else result


def get_cache_folder(cache_folder: str=None):
    if cache_folder is None:
        cache_folder = os.getenv('ZERONLG_HOME', None) or os.getenv('CKPT_HOME', None)
        if cache_folder is None:
            try:
                from torch.hub import _get_torch_home

                torch_cache_home = _get_torch_home()
            except ImportError:
                torch_cache_home = os.path.expanduser(os.getenv('TORCH_HOME', os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))

            cache_folder = os.path.join(torch_cache_home, 'zeronlg')
    return cache_folder


def get_formatted_string(kwargs, key, assigned_keys=None, assigned_kwargs=None, format_key=None) -> str:
    if format_key is None:
        format_key = f'{key}_format'

    if kwargs.get(key, None) is None:
        assert format_key in kwargs
        
        if assigned_kwargs is None:
            assert assigned_keys is not None
            assigned_kwargs = {k: kwargs[k] for k in assigned_keys}

        return kwargs[format_key].format(**assigned_kwargs)

    return kwargs[key]


def download_if_necessary(
        model_name_or_path: str, 
        cache_folder: str = None, 
        use_auth_token: bool = False,
        key_files: str = ['config.json', 'pytorch_model.bin'],
        ignore_files: List[str] = ['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5', 'model.onnx', 'model.safetensors'],
    ):

    assert model_name_or_path is not None

    if cache_folder is None:
        cache_folder = get_cache_folder()

    if os.path.exists(model_name_or_path):
        model_path = model_name_or_path
    else:
        model_path = os.path.join(cache_folder, model_name_or_path.replace('/', '_'))
    
    if any(not os.path.exists(os.path.join(model_path, key_file)) for key_file in key_files):
        storage_path = snapshot_download(model_name_or_path,
                        cache_dir=cache_folder,
                        library_name=__LIBRARY_NAME__,
                        library_version=__version__,
                        ignore_files=ignore_files,
                        use_auth_token=use_auth_token)
        assert model_path == storage_path
    
    assert os.path.exists(model_path)
    return model_path
