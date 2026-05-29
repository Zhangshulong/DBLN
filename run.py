import os
import argparse
import time
import torch
import csv
import torch.nn as nn
from utils.ls_utils import print_args, set_seed, build_loss, build_optimizer,evaluate_model,save_cls_result, build_model,create_result_csv,save_result_csv
from data_preprocess.data_loders import load_data, data_loaders
from utils.dict_patch import uea_dict_patch
def build_args():
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--task_name', type=str, default='classification',help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model', type=str,  default='DBLN', help='model name, options: [DBLN, SoftShapeNet, TSLANet, Medformer, ModernTCN, UniTS')
    
    # DBLN
    parser.add_argument('--decomp', type=int, default = 1, help='1:True , 0:False')
    parser.add_argument('--hvmamba', type=int, default = 1, help='1:True , 0:False')
    parser.add_argument('--fuse',   type=int, default = 1, help='1:True , 0:False')
    parser.add_argument('--use_patch_dict', type=int, default = 0, help='1:True , 0:False')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--ls_scale', type=int, default=2, help='long-short scale factor')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--d_state', type=int, default=16, help='SSM state expansion factor')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--pre_norm',   type=int, default = 1, help='1:True , 0:False')
    
    # data loader
    # Dataset setup
    parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
    parser.add_argument('--datatype', type=str, default= 'UEA', help= 'UEA')
    parser.add_argument('--dataset', type=str, default = 'BasicMotions', help= 'BasicMotions')

    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')

    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=3, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    parser.add_argument('--c_out', type=int, default=7, help='output size for MICN')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',help='whether to use distilling in encoder, using this argument means not using distilling',default=True)
    parser.add_argument('--embed', type=str, default='patch',help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=1, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96, help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=4, help='for Inception')

    # optimization
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--loss_fun', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--epoch', type=int, default=200, help='train epochs')
    parser.add_argument('--warm_up_epoch', type=int, default=100, help='50, 100 or 200')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--lr', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default = 0, help='weight decay')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    # parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # SoftShapeNet
    parser.add_argument('--sparse_rate', type=float, default=0.50, help='0.1, 0.3, or 0.7')
    parser.add_argument('--moe_num_experts', type=int, default=8)

    # Medformer
    parser.add_argument("--patch_len_list",type=str,default="4,8,16",help="a list of patch len used in Medformer",)
    parser.add_argument("--single_channel",action="store_true",help="whether to use single channel patching for Medformer",default=False,)
    parser.add_argument("--augmentations",type=str,default="flip,frequency,jitter,mask,channel,drop",help="A comma-seperated list of augmentation types (none, jitter or scale). ""Randomly applied to each granularity. ""Append numbers to specify the strength of the augmentation, e.g., jitter0.1",)
    parser.add_argument("--output_attention",action="store_true",help="whether to output attention in encoder",)
    parser.add_argument("--no_inter_attn",action="store_true",help="whether to use inter-attention in encoder, using this argument means not using inter-attention",default=False,)
    
    # ModernTCN
    parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
    parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio')
    parser.add_argument('--patch_size', type=int, default=16, help='the patch size')
    parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')
    parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1,1], help='num_blocks in each stage')
    parser.add_argument('--large_size', nargs='+',type=int, default=[31,29,27,13], help='big kernel size')
    parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5,5], help='small kernel size for structral reparam')
    parser.add_argument('--dims', nargs='+',type=int, default=[256,256,256,256], help='dmodels in each stage')
    parser.add_argument('--dw_dims', nargs='+',type=int, default=[256,256,256,256], help='dw dims in dw conv in each stage')
    parser.add_argument('--small_kernel_merged', type=bool, default=False, help='small_kernel has already merged or not')
    parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
    parser.add_argument('--use_multi_scale', type=bool, default=False, help='use_multi_scale fusion')
    
    # classfication task
    parser.add_argument('--class_dropout', type=float, default=0.05, help='classfication dropout')

    # UniTS
    parser.add_argument("--prompt_num", type=int, default=5)

    # UniShape
    parser.add_argument('--input_size', type=int, default=1, help='input_size')
    parser.add_argument('--in_channels', type=int, default=128)
    parser.add_argument('--window_emb_dim', type=int, default=128)
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--shape_ratio', type=float, default=0.6)
    parser.add_argument('--scale_len', type=int, default=5) ## 1， 2， 3， 4， 5
    parser.add_argument('--ensemble_num_model', type=int, default=5) ## 1
    parser.add_argument('--pretrain_label_ratio', type=float, default=0.1, help='')

    # TimeMixer
    parser.add_argument('--use_future_temporal_feature', type=int, default=0,help='whether to use future_temporal_feature; True 1 False 0')

    # Result setup
    parser.add_argument('--checkpoints', type=str, default='./exp/checkpoints', help='location of model checkpoints')
    parser.add_argument('--results', type=str, default='./exp/results')

    # GPU
    parser.add_argument('--use_multi_gpu', type=int, default=0, help='use multiple gpus')
    parser.add_argument('--device', type=str, default='cuda:1', help='use cuda device')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    '''--------------------------------build params------------------'''
    args = build_args()
    args.decomp = True if args.decomp ==1 else False
    args.hvmamba = True if args.hvmamba ==1 else False
    args.fuse = False if args.decomp == False else True if args.fuse ==1 else False
    set_seed(args.seed)
    '''--------------------------------save model setting-------------------'''
    checkpoints_setting = '{}_{}_{}_d{}_l{}_lr{}'.format(args.model, args.datatype, args.dataset, args.d_model, args.e_layers, args.lr)
    if args.model == 'DBLN':
        if args.use_patch_dict==1:
            args.patch_len = uea_dict_patch[args.dataset]['patch_len']
            args.ls_scale = uea_dict_patch[args.dataset]['ls_scale']
        checkpoints_setting += '_dec{}_hvm{}_f{}_p{}_s{}'.format(args.decomp, args.hvmamba, args.fuse, args.patch_len, args.ls_scale)
    # save checkpoints
    checkpoints_name = checkpoints_setting + '.pth'
    checkpoints_path =  args.checkpoints + '/' + args.datatype + '/' + args.model
    checkpoints = os.path.join(checkpoints_path, checkpoints_name)
    os.makedirs(os.path.dirname(checkpoints), exist_ok=True)
    # save results
    results_setting = '{}_{}_d{}_l{}_lr{}'.format(args.model, args.datatype, args.d_model, args.e_layers, args.lr)
    results_name = results_setting + '.csv'
    results_path = args.results + '/' + args.datatype
    results = os.path.join(results_path, results_name)
    os.makedirs(os.path.dirname(results), exist_ok=True)
    create_result_csv(results, args.dataset, name=args.model)

    '''--------------------------------loading data----------------'''
    sum_dataset, sum_target, args.num_class = load_data(args.root_path, args.datatype, args.dataset)
    args.enc_in, args.seq_len = sum_dataset.shape[1], sum_dataset.shape[2]
    # adjust patch_len and label_len
    if args.patch_len >= args.seq_len:
        args.patch_len = args.seq_len
    if args.task_name == 'classification':
        args.pred_len = 0
    args.label_len = args.seq_len // 2

    '''--------------------------------build_model------------------'''
    model = build_model(args)
    args.params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    '''--------------------------build_optimizer_and_loss-----------'''
    loss_fun = build_loss(args.loss_fun)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    '''--------------------------------GPU setting-----------------'''
    if args.use_multi_gpu==1 and torch.cuda.is_available():
        device = [torch.device(f'cuda:{i}') for i in range(2)]
        model = nn.DataParallel(model, device_ids=device)
    
    # print model and dataset setting
    model_rows = [("is_training", args.is_training), ("use_datatype", args.datatype), ("use_dataset", args.dataset), 
                  ("shape_dataset", sum_dataset.shape),("num_classes", args.num_class), ("batch_size", args.batch_size),
                  ("use_model", args.model), ("d_model", args.d_model), ("depth", args.e_layers), ("total_params", args.params),
                  ("checkpoint", checkpoints), ("results", results), ("device", args.device)]
    print_args(model_rows) # if args.is_training==1 else None

    '''--------------------------------model training-------------'''
    # data loaders
    train_loader, val_loader, test_loader = data_loaders(args.batch_size, sum_dataset, sum_target,args.seed)
    if args.is_training==1:
        model = model.to(args.device)
        use_time = 0
        t = time.time()
        last_val_loss, min_val_loss = float('inf'), float('inf')
        stop_count, increase_count = 0, 0
        for epoch in range(args.epoch):
            model.train()
            train_loss = 0
            num_iterations = 0
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = x.to(args.device), y.to(args.device)
                x_mark = None
                outputs = model(x, x_mark, None,None)
                loss = loss_fun(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss.item()
                num_iterations = num_iterations + 1 
            train_loss = train_loss / num_iterations

            model.eval()
            val_loss, val_acc = evaluate_model(val_loader, model, loss_fun, args.device)

            if (epoch > args.warm_up_epoch) and (val_loss > min_val_loss):
                increase_count =  increase_count + 1
            else:
                increase_count = 0

            if min_val_loss > val_loss:
                torch.save(model.state_dict(), checkpoints)
                min_val_loss = val_loss
                test_loss, test_acc = evaluate_model(test_loader, model, loss_fun, args.device)
                print('epoch {}, train_loss {:.4f} val_loss {:.4f} val_acc {:.4f} test_acc {:.4f} *'.format(epoch+1, train_loss, val_loss, val_acc, test_acc))
                
            if (epoch+1) % 100 == 0:
                print('epoch {}, train_loss {:.4f} val_loss {:.4f} val_acc {:.4f} test_acc {:.4f} *'.format(epoch+1, train_loss, val_loss, val_acc, test_acc))

            if (epoch > args.warm_up_epoch) and (abs(last_val_loss - val_loss) <= 1e-4):
                stop_count = stop_count + 1
            else:
                stop_count = 0

            if stop_count == args.patience or increase_count == args.patience:
                print('model convergent at epoch {}, early stopping'.format(epoch))
                break
            last_val_loss = val_loss
        t = time.time() - t
        use_time += t

    else:
        use_time = 0
        t = time.time()
        # load model
        model.load_state_dict(torch.load(checkpoints))
        model = model.to(args.device)
        model.eval()
        test_loss, test_acc = evaluate_model(test_loader, model, loss_fun, args.device)
        t = time.time() - t
        use_time += t
    torch.cuda.empty_cache() # clear GPU cache

    # print data-model information
    model_rows = [("is_training", args.is_training), ("use_datatype", args.datatype), ("use_dataset", args.dataset), 
                  ("shape_dataset", sum_dataset.shape),("num_classes", args.num_class), ("batch_size", args.batch_size),
                  ("use_model", args.model), ("d_model", args.d_model), ("depth", args.e_layers), ("total_params", args.params),
                  ("patch_len", args.patch_len), ("ls_scale", args.ls_scale), ("checkpoint", checkpoints), ("results", results),
                  ("device", args.device),("use_time", use_time),("accuracy", test_acc)]
    print_args(model_rows)

    # save model results
    save_result_csv(results, args.dataset, test_acc)