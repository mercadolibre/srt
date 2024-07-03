import traceback
from multiprocessing import Pool, cpu_count
from pprint import pprint

from src import fs
from src.progress import progress


def process_files(in_dir, func, processes=-1, maxtasksperchild=1, desc=None, **common_kwargs):
    fnames = fs.glob(fs.join(in_dir, '*.jl.gz'))
    return process_file_list(
        fnames, func, processes=processes, maxtasksperchild=maxtasksperchild, desc=desc, **common_kwargs
    )


def process_file_list(fnames, func, processes=-1, maxtasksperchild=1,
                      results_iterator=False, desc=None, print_progress=True, keep_input=True, **common_kwargs):
    it = _process_file_list(
        fnames, func, processes=processes, maxtasksperchild=maxtasksperchild,
        desc=desc, print_progress=print_progress, keep_input=keep_input, **common_kwargs
    )
    if results_iterator:
        return it
    else:
        return list(it)


def _process_file_list(inputs, func, processes=-1, maxtasksperchild=1, desc=None, print_progress=True,
                       keep_input=True, **common_kwargs):
    jobs = []
    for job_input in inputs:
        job = common_kwargs.copy()
        job['func'] = func
        job['input'] = job_input
        job['keep_input'] = keep_input
        jobs.append(job)

    desc = desc or f'compute of {func.__name__}'
    failed = []
    if processes == -1: processes = cpu_count()
    if processes > 1:
        with Pool(processes, maxtasksperchild=maxtasksperchild) as p:
            yield from _process_file_list_inner(desc, failed, jobs, p.imap_unordered, print_progress=print_progress)
    else:
        yield from _process_file_list_inner(desc, failed, jobs, map, print_progress=print_progress)

    if failed:
        print(f'Some jobs failed computing {func.__name__}')
        print("*" * 20)
        print()
        for result in failed:
            if keep_input: pprint(result['job'])
            print(result['traceback'])
            print('\n' + ('*' * 20) + '\n')

        raise RuntimeError('Some slices failed')


def _process_file_list_inner(desc, failed, jobs, map_func, *, print_progress=False):
    it = map_func(_do_job, jobs)
    if print_progress:
        it = progress(it, tot=len(jobs), desc=desc, start_position=0)

    for result in it:
        if result['status'] != 'ok':
            failed.append(result)
            break
        yield result


def _do_job(job):
    func = job.pop('func')
    try:
        res = func(job['input'], **{k: v for k, v in job.items() if k != 'input' and k != 'keep_input'})
        res = dict(status='ok', res=res)
        if job['keep_input']:
            res['job'] = job
        return res
    except:
        traceback.print_exc()
        res = dict(status='failed', traceback=traceback.format_exc())
        if job['keep_input']:
            res['job'] = job
        return res

