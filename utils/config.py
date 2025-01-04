import yaml


class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.teacher_model_name_or_path = cfg['teacher_model_name_or_path']
        self.student_model_name_or_path = cfg['student_model_name_or_path']
        self.filter_architecture = cfg['filter_architecture']
        self.num_layers = cfg['num_layers']
        self.layer_mapping_strategy = cfg['layer_mapping']
        self.stage1_epochs = cfg['stage1_epochs']
        self.stage2_epochs = cfg['stage2_epochs']
        self.learning_rate = cfg['learning_rate']
        self.batch_size = cfg['batch_size']
        self.alpha1 = cfg['alpha1']
        self.alpha2 = cfg['alpha2']
        self.temperature = cfg['temperature']
        self.mixed_precision = cfg['mixed_precision']
        self.output_dir = cfg['output_dir']
        self.task_name = cfg['task_name']
        self.max_length = cfg['max_length']
        self.pad_to_max_length = cfg['pad_to_max_length']
        self.weight_decay = cfg['weight_decay']
        self.gradient_accumulation_steps = cfg['gradient_accumulation_steps']
        self.lr_scheduler_type = cfg['lr_scheduler_type']
        self.num_warmup_steps = cfg['num_warmup_steps']

        # 其他配置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task_output_size = cfg.get('task_output_size', 2)  # 根据任务调整
        self.teacher_hidden_size = cfg.get('teacher_hidden_size', 768)
        self.student_hidden_size = cfg.get('student_hidden_size', 384)
