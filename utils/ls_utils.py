import random
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import csv

from models import LongShort
def create_result_csv(file_path, dataset, name="Accuracy"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Dataset', name])
        print(f"Result CSV file created at: {file_path}")

    with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            existing_datasets = [row[0] for row in reader if row]
            if dataset not in existing_datasets:
                with open(file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([dataset, ''])
    return file_path

def save_result_csv(file_path, dataset, accuracy):
    rows = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            row[1] = accuracy if row[0] == dataset else row[1]
            rows.append(row)
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
    print(f"Save Accuracy for {dataset} in {file_path}")

def set_seed(random_seed):
    random.seed(random_seed)                       
    os.environ['PYTHONHASHSEED'] = str(random_seed) 
    np.random.seed(random_seed)                    
    torch.manual_seed(random_seed)                 
    torch.cuda.manual_seed(random_seed)            
    torch.cuda.manual_seed_all(random_seed)        
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     

def print_args(rows):
    col1w = max(len(r[0]) for r in rows) + 2
    col2w = max(len(str(r[1])) for r in rows) + 2

    print("+" + "-" * col1w + "+" + "-" * col2w + "+")
    print(f"| {'Name':<{col1w-2}} | {'Value':<{col2w-2}} |")
    print("+" + "-" * col1w + "+" + "-" * col2w + "+")
    for k, v in rows:
        print(f"| {k:<{col1w-2}} | {str(v):<{col2w-2}} |")
    print("+" + "-" * col1w + "+" + "-" * col2w + "+")


def build_loss(loss_fun):
    if loss_fun == 'cross_entropy':
        loss_fun = nn.CrossEntropyLoss()
    elif loss_fun == 'reconstruction':
        loss_fun = nn.MSELoss()
    return loss_fun

def build_optimizer(model, optimizer, lr, weight_decay):
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def evaluate_model(val_loader, net, loss_fun, device):
    val_loss, val_acc = 0, 0
    sum_len = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            x_mark = None
            outputs = net(x,x_mark,None,None)
            val_loss = val_loss + loss_fun(outputs, y).item()
            acc = (outputs.argmax(1) == y).sum().item()
            val_acc = val_acc + acc
            sum_len = sum_len + len(y)
    avg_val_loss = val_loss / (i+1)
    #print(val_acc, sum_len)
    avg_val_acc = val_acc / sum_len

    return avg_val_loss, avg_val_acc

def flops_model(net, args):
    from thop import profile
    import copy
    net_copy = copy.deepcopy(net) # 复制模型
    device = args.device
    input = torch.randn(32, args.seq_len,args.enc_in).to(device)
    flops, params = profile(net_copy.to(device), inputs=(input, None, None, None))
    return flops/1e9, params/1e6

# 利用混淆矩阵计算Precision, Recall, F1score
# def result_tpfpfn(result, label, args):
#     from torchmetrics import ConfusionMatrix
#     confmat = ConfusionMatrix(task="multiclass", num_classes=args.num_classes).to(args.device)
#     confmat_result = confmat(result.argmax(1), label)
#     # 计算每个类别的TP, FP, FN
#     TP = confmat_result.diag()  # 对角线元素为TP
#     FP = confmat_result.sum(dim=0) - TP  # 列和减去TP
#     FN = confmat_result.sum(dim=1) - TP  # 行和减去TP
#     # 计算每个类别的Accuary,Precision,Recall,F1-score
#     accuary_per_class = TP / confmat_result.sum(dim=1)
#     precision_per_class = TP / (TP + FP )
#     recall_per_class = TP / (TP + FN )
#     f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
#     # 计算宏平均（macro average）
#     Precision = torch.mean(precision_per_class) * 100
#     Recall = torch.mean(recall_per_class) * 100
#     F1score = torch.mean(f1_per_class) * 100
#     args.confmat_result = confmat_result

#     return Precision, Recall, F1score

# 计算模型多分类ROC曲线AUC值, Top-1, Top-2准确率
def roc_auc(result, label, args):
    from sklearn.metrics import roc_auc_score,top_k_accuracy_score
    import torch.nn.functional as F
    label_cpu = label.cpu().numpy()
    result_probs = F.softmax(result, dim=1)  # 获取概率分数（softmax输出）
    result_cpu = result_probs.cpu().numpy()  # 使用概率分数而不是预测标签
    if args.num_classes > 2:
        roc_auc = roc_auc_score(label_cpu, result_cpu, multi_class='ovr', average='macro')
        # 计算top-k准确率
        top1_acc = top_k_accuracy_score(label_cpu, result_cpu, k=1) * 100
        top2_acc = top_k_accuracy_score(label_cpu, result_cpu, k=2) * 100
    else: # 二分类
        top1_acc = ((result.argmax(1) == label).sum().item() / len(label)) * 100
        top2_acc = 100.0
    return roc_auc, top1_acc, top2_acc

def save_cls_result(args, mean_accu, train_time):
    args.save_csv_name = 'results'
    save_path = os.path.join(args.save_dir, '', args.save_csv_name + '_cls.csv')
    if os.path.exists(save_path):
        result_form = pd.read_csv(save_path, index_col=0)
    else:
        result_form = pd.DataFrame(columns=['dataset_name', 'mean_accu', 'train_time'])

    result_form = pd.concat([result_form, pd.DataFrame([{'dataset_name': args.dataset, 'mean_accu': '%.4f' % mean_accu, 'train_time': '%.4f' % train_time}])], ignore_index=True)

    result_form.to_csv(save_path, index=True, index_label="id")


from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, KANAD, TSLANet,SoftShapeNet,Medformer,ModernTCN,UniTS,LongShort,MILLET,InterpGN,\
    TimeCHEAT,UniShape,TEFN,LongShort1,LongShort2,LongShort3,TMNet

model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'KANAD': KANAD,
            'TSLANet': TSLANet,  # for backward compatibility
            'SoftShapeNet': SoftShapeNet,
            'LongShort': LongShort,
            'Medformer': Medformer,
            'ModernTCN': ModernTCN,
            'UniTS': UniTS,
            'MILLET': MILLET,
            'InterpGN': InterpGN,
            'TimeCHEAT': TimeCHEAT,
            'UniShape': UniShape,
            'TEFN': TEFN,
            'LongShort1': LongShort1,
            'LongShort2': LongShort2,
            'LongShort3': LongShort3,
            'TMNet': TMNet,
        }
def build_model(args):
    model = model_dict[args.model].Model(args).float()
    return model

def get_save_path(args):
    if args.datatype == 'Wave':
        path_log = f'{args.save_dir}/logdirs/{args.datatype}/{args.dataset}/{args.model}/{args.timesize}'
        save_path = f'{args.save_dir}/save_model/{args.datatype}/{args.dataset}/{args.model}/{args.timesize}'
    else:
        path_log = f'{args.save_dir}/logdirs/{args.datatype}/{args.dataset}/{args.model}'
        save_path = f'{args.save_dir}/save_model/{args.datatype}/{args.dataset}/{args.model}'
    return path_log, save_path