from src.fda_utils.api_client import FDAClient
from src.conf_utils import ConfigManager
from src.utils import dict2commandline

default_overrides = dict(
    epochs=4,
    train_batch_size=512, eval_batch_size=512,
    scheduler={'type': 'one-cycle'}, stopping_step=20, eval_step=2,
    nce_temperature=1,
    nce_label_smoothing=0.0,
    nce_global_negatives=True,
    nce_sampling_strategy='popularity',
    nce_num_negatives=1000,
    save_dataset=True,
    inner_size=64, hidden_size=64,

)




def schedule_pretrain(pre_train_version, dataset, config_dict, out_version_str, tags=None):
    client = FDAClient('personalization-research')

    outputs_versions = {
        'run_metrics': out_version_str,
        'run_model': out_version_str,
        'config_dict': out_version_str,
        # this goes for version 0.1.X
        'partial_metrics': out_version_str,
    }

    dt_name, dt_version = dataset.split('==')
    dataset_id = client.get_by_name_version(dt_name, dt_version)['visible_id']
    return client.run_task(
        'train_sasrecf2', pre_train_version,
        params={'config_dict': dict2commandline(config_dict), 'model': 'SASRecF2'},
        artifacts=[{'alias': 'meli_purchs', 'id': dataset_id}],
        outputs_versions=outputs_versions,
        flavor='new_gpu',
        tags=tags
    )


def test_run(out_version_str='0.0.1-test', pre_train_version='0.0.37-oldcuda'):
    model = 'SASRecF2'
    dataset = 'meli_purchs==0.0.3-800k'
    dt_name, dt_version = dataset.split('==')
    loss_type = 'InfoNCE-quick'
    epochs = 5
    outputs_versions = {'run_metrics': out_version_str, 'run_model': out_version_str}

    config_dict = get_config_dict(dt_name, model, epochs, loss_type)
    return schedule_pretrain(pre_train_version, dataset, config_dict, outputs_versions)
