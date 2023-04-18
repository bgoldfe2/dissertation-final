class Model_Config:

    def __init__(self, args):
        #parser = get_parser()
        #args = parser.parse_args()

        self.max_length=args.max_length         
        self.train_batch_size=args.train_batch_size    
        self.valid_batch_size=args.valid_batch_size     
        self.test_batch_size=args.test_batch_size      
        self.epochs=args.epochs
        self.learning_rate=args.learning_rate
        self.weight_decay=args.weight_decay
        self.adamw_epsilon=args.adamw_epsilon
        self.warmup_steps=args.warmup_steps
        self.classes=args.classes
        self.dropout=args.dropout
        self.seed=args.seed
        self.device=args.device
        self.dataset=args.dataset

        self.pretrained_model=args.pretrained_model
        self.deberta_hidden=args.deberta_hidden
        self.gpt_neo_hidden=args.gpt_neo_hidden
        self.gpt_neo13_hidden=args.gpt_neo13_hidden
        self.roberta_hidden=args.roberta_hidden
        self.xlnet_hidden=args.xlnet_hidden
        self.albert_hidden=args.albert_hidden
        self.ensemble_type=args.ensemble_type

        self.run_path=args.run_path
        self.dataset_path=args.dataset_path
        self.model_path=args.model_path
        self.output_path=args.output_path
        self.figure_path=args.figure_path
        self.split=args.split

        self.model_list = None
        #self.current_model = None