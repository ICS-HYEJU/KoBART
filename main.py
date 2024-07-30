from transformers import PreTrainedTokenizerFast

from data.dataloader import KobartSummaryDataModule
from model.bart import KoBARTGeneration

import torch

if __name__ == '__main__':
    from config.config import get_config_dict
    cfg = get_config_dict()
    #
    if cfg.device['gpu_id'] is not None:
        device = torch.device('cuda:{}'.format(cfg.device['gpu_id']))
        torch.cuda.set_device(cfg.device['gpu_id'])
    else:
        device = torch.device('cpu')
    ##############Engine###################
    dm = KobartSummaryDataModule(
        train_file=cfg.dataset_info['train_file'],
        test_file=cfg.dataset_info['test_file'],
        max_len =cfg.dataset_info['max_len'],
        batch_size=cfg.dataset_info['batch_size'],
        num_workers=0,
        pretrained_name='gogamza/kobart-base-v1'
    )
    dm.setup('fit')
    #
    tok = PreTrainedTokenizerFast.from_pretrained(cfg.dataset_info['pretrained_name'])
    model = KoBARTGeneration(cfg, tok=tok).to(device)
    #
    for batch_idx, data in enumerate(dm.train_dataloader()):
        data_d = dict()
        data_d['input_ids'] = data['input_ids'].to(device)
        data_d['dec_input_ids'] = data['dec_input_ids'].to(device)
        data_d['label_ids'] = data['label_ids'].to(device)

        out = model(data_d)
        out.keys()
