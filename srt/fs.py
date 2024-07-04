import gzip
import os
import re
import shutil
from glob import glob as local_glob
from pathlib import Path
from urllib.parse import urlparse

from melitk.melipass import get_secret


# import boto3
# import s3fs


# Variables remote_fs and s3_client are instantiated inside the functions because there are problems on sharing them across threads
# See https://github.com/dask/dask/issues/1292#issuecomment-226293446
def get_s3_session():
    return boto3.Session(
        aws_access_key_id=get_secret('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=get_secret('AWS_SECRET_KEY'),
    )


def get_s3_client():
    s3_client = get_s3_session().client('s3')
    return s3_client


def get_s3fs(default_cache_type='bytes'):
    remote_fs = s3fs.S3FileSystem(
        key=get_secret('AWS_ACCESS_KEY_ID'),
        secret=get_secret('AWS_SECRET_KEY'),
        default_cache_type=default_cache_type
    )
    return remote_fs


def smart_open(fname, mode='r', compressed=None, prefetch=False):
    if compressed is None:
        compressed = is_gz(fname)

    if is_local(fname):
        return gzip.open(fname, mode) if compressed else open(fname, mode)
    else:
        if compressed and not mode.endswith('b'):
            mode = mode + 'b'

        remote_fs = get_s3fs(default_cache_type='readahead')
        remote_fs.invalidate_cache()  # tell me why da fac this caches things

        res = remote_fs.open(fname, mode)  # , block_size=50 * 2 ** 20)
        if prefetch:
            res.cache._fetch(0, 10)

        if compressed:
            res = gzip.GzipFile(fileobj=res, mode=mode)
        return res


def exists(fname):
    if is_local(fname):
        return Path(fname).exists()
    else:
        remote_fs = get_s3fs()
        remote_fs.invalidate_cache()  # tell me why da fac this caches things
        return remote_fs.exists(fname)


def join(path, *dirs):
    return os.path.join(path, *dirs)


def mkdir(path, parents=False, exist_ok=False):
    if is_local(path):
        return Path(path).mkdir(parents=parents, exist_ok=exist_ok)
    else:
        return  # no need on s3


def change_ext(path, ext):
    return re.sub('(\.\w{1,3})+$', ext, path)


def clear_ext(path):
    return change_ext(path, '')


def abspath(path):
    if is_local(path):
        return os.path.abspath(path)
    else:
        return path


def ensure_exists(path, clean=False):
    if clean:
        rmtree(path, not_exist_ok=True)
    mkdir(path, parents=True, exist_ok=True)
    return path


def ensure_not_exists(path):
    if exists(path):
        if is_dir(path):
            rmtree(path)
        else:
            remove(path)
    return path


def ensure_clean(path):
    return ensure_exists(path, clean=True)


def move(src, dst):
    if is_local(src) and is_local(dst):
        shutil.move(src, dst)
    elif not is_local(src) and not is_local(dst):
        remote_fs = get_s3fs()
        remote_fs.move(src, dst)
    elif is_local(src) and not is_local(dst):
        s3_client = get_s3_client()
        bucket, key = _parse_s3_path(dst)
        with open(src, 'rb') as f:
            s3_client.put_object(
                Body=f.read(),
                Bucket=bucket,
                Key=key
            )
        os.unlink(src)
    else:  # not is_local(src) and is_local(dst)
        bucket, key = _parse_s3_path(src)
        s3_client = get_s3_client()
        s3_client.download_file(bucket, key, dst)
        s3_client.rm(src)


def copy(src, dst):
    if is_local(src) and is_local(dst):
        shutil.copy(src, dst)
    elif not is_local(src) and not is_local(dst):
        remote_fs = get_s3fs()
        remote_fs.copy(src, dst)
    elif is_local(src) and not is_local(dst):
        s3_client = get_s3_client()
        bucket, key = _parse_s3_path(dst)
        with open(src, 'rb') as f:
            s3_client.put_object(
                Body=f.read(),
                Bucket=bucket,
                Key=key
            )
    else:  # not is_local(src) and is_local(dst)
        s3_client = get_s3_client()
        bucket, key = _parse_s3_path(src)
        s3_client.download_file(bucket, key, dst)


def fast_copy(src, dst):
    rec = '--recursive ' if is_dir(src) else ''
    os.system(f'aws s3 cp --quiet {rec}"{src}" "{dst}"')
    # Popen(shlex.split(f'aws s3 cp {rec}"{src}" "{dst}"'), stdout=-1).wait()


def rmtree(path, not_exist_ok=False):
    if not_exist_ok and not exists(path): return

    if is_local(path):
        return shutil.rmtree(path)
    else:
        remote_fs = get_s3fs()
        remote_fs.invalidate_cache()
        return remote_fs.rm(path, recursive=True)


def remove(path, not_exist_ok=False):
    if not_exist_ok and not exists(path): return

    if is_local(path):
        return os.unlink(path)
    else:
        remote_fs = get_s3fs()
        remote_fs.invalidate_cache()
        return remote_fs.rm(path, recursive=False)


def glob(pattern):
    if is_local(pattern):
        return sorted(local_glob(pattern))
    else:
        remote_fs = get_s3fs()
        remote_fs.invalidate_cache()  # tell me why da fac this caches things
        return [f's3://{e}' for e in sorted(remote_fs.glob(pattern))]


def walk_files(path):
    """
    Does not return sub directories
    """
    if is_local(path):
        for root, subdirs, fnames in os.walk(path):
            for fname in fnames:
                yield join(root, fname)
    else:
        remote_fs = get_s3fs()
        remote_fs.invalidate_cache()  # tell me why da fac this caches things
        for e in remote_fs.find(path):
            yield f's3://{e}'


def touch(fname):
    with smart_open(fname, 'w'): pass


def is_dir(fname):
    if is_local(fname):
        return os.path.isdir(fname)
    else:
        return len(ls(fname)) > 0


def is_file(fname):
    return exists(fname) and not is_dir(fname)


def ls(*path):
    path = join(*path)
    return glob(join(path, '*'))


def name(fname):
    return fname.rstrip('/').split('/')[-1]


def _parse_s3_path(path):
    parsed = urlparse(path, allow_fragments=False)
    return parsed.netloc, parsed.path[1:]


def is_local(fname):
    return not fname.startswith('s3://')


def is_gz(fname):
    return fname.endswith('.gz')


def strip_ext(name):
    return name.split('.')[0]


def get_ext(fname):
    return '.'.join(name(fname).split('.')[1:])


def parent(path, n=1):
    res = path
    for _ in range(n):
        res = os.path.dirname(res)
    return res
