import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor.transform import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, ToTensor, ImageNormalize
from jittor.models import Resnet50
from tqdm import tqdm
import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import argparse
import re
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter, defaultdict
from jimm.models import swin_base_patch4_window7_224_in22k
import random
import pandas as pd
from jittor.lr_scheduler import CosineAnnealingLR
import albumentations as A
from PIL import ImageOps
from typing import Optional, List, Tuple, Union, Dict
import uuid

jt.flags.use_cuda = 1

# ============== Weight ==============
class DynamicWeighting:
    def __init__(self, num_classes):
        self.class_weights = jt.ones(num_classes, dtype=jt.float32)
        self.val_confusion = None

    def update(self, val_conf_matrix):
        # 计算各类别的错误率（0~1之间）
        error_rates = 1 - val_conf_matrix.diagonal() / (val_conf_matrix.sum(axis=1) + 1e-6)

        # 将错误率映射到权重范围（如1.0~5.0）
        min_weight, max_weight = 1.0, 5.0
        weights = min_weight + (max_weight - min_weight) * error_rates

        # 确保数值稳定性
        self.class_weights = jt.array(weights, dtype=jt.float32)
        print(f"Updated weights: {self.class_weights.numpy()}")

# ============== Replace ==============
class HardSampleBank:
    def __init__(self, capacity=500):
        self.bank = []
        self.capacity = capacity

    def add(self, images, labels, preds, probs):
        """添加验证集中分类错误的难样本"""
        wrong_mask = (preds != labels)
        hard_mask = (probs.max(axis=1) < 0.6)  # 置信度低于0.6设置为难样本

        for img, lbl in zip(images[wrong_mask & hard_mask], labels[wrong_mask & hard_mask]):
            if len(self.bank) < self.capacity:
                self.bank.append((img, lbl))
            else:
                # 随机替换
                idx = random.randint(0, self.capacity-1)
                self.bank[idx] = (img, lbl)

    def get_batch(self, batch_size):
        """获取一批难样本"""
        if len(self.bank) == 0:
            return None
        indices = np.random.choice(len(self.bank), min(batch_size, len(self.bank)))
        batch = [self.bank[i] for i in indices]
        return jt.stack([x[0] for x in batch]), jt.array([x[1] for x in batch])

class HardClassAugment:
    """针对难样本的增强"""
    def __call__(self, img, label):
        if label in [2,3,4]:  # 只对难样本增强
            if random.random() > 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,1.5)))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7,1.3))
            if random.random() > 0.5:
                img = img.rotate(random.randint(-15,15))
        return img

# ============== Dataset ==============
class ImageFolder(Dataset):
    def __init__(self, root: str,
                 annotation_path: Optional[str] = None,
                 transform: Optional[Compose] = None,
                 oversample_minority: bool = False,
                 hard_augment: bool = False,
                 is_train: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.transform = transform
        self.oversample_minority = oversample_minority
        self.hard_augment = hard_augment
        self.is_train = is_train
        self.hard_augmentor = HardClassAugment() if hard_augment else None

        # 增强配置
        self.alb_transform = A.Compose([
            A.CLAHE(p=0.5),
            A.GaussianBlur(p=0.3),
            A.RandomGamma(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ]) if is_train else None

        # 加载数据
        self.data_dir = self._load_data(annotation_path)
        self._setup_class_balance()

        if self.oversample_minority and self.label_counts is not None:
            self.data_dir = self._balance_data_with_augmentation()
            self._print_class_distribution("After Balanced Augmentation")

        self.total_len = len(self.data_dir)

    def _load_data(self, annotation_path: Optional[str]) -> List[Tuple[str, Optional[int]]]:
        if annotation_path:
            with open(annotation_path) as f:
                return [(x[0], int(x[1])) for x in (line.strip().split() for line in f)]
        else:
            return [(x, None) for x in sorted(os.listdir(self.root)) if not x.startswith('aug_')]

    def _setup_class_balance(self):
        if any(label is None for _, label in self.data_dir):
            self.label_counts = None
            self.minority_class = None
        else:
            self.label_counts = Counter(label for _, label in self.data_dir)
            self.minority_class = min(self.label_counts, key=self.label_counts.get)

    def _balance_data_with_augmentation(self) -> List[Tuple[Union[str, np.ndarray], int]]:
        """使用Albumentations进行智能过采样"""
        class_samples = defaultdict(list)
        for img, label in self.data_dir:
            class_samples[label].append(img)

        max_count = max(len(v) for v in class_samples.values())
        balanced_data = []

        for label, samples in class_samples.items():
            balanced_data.extend([(img, label) for img in samples])
            need_count = max_count - len(samples)

            if need_count > 0:
                class_aug = self._get_class_augmentor(label)
                balanced_data.extend(self._generate_augmented_samples(
                    samples, label, need_count, class_aug))

        random.shuffle(balanced_data)
        return balanced_data

    def _get_class_augmentor(self, label: int) -> A.Compose:
        """根据类别返回不同的增强策略"""
        base_aug = [
            A.Rotate(limit=45, p=0.7),
            A.RandomBrightnessContrast(p=0.5),
            A.GridDistortion(p=0.3),
        ]
        if label == self.minority_class:  # 对少数类使用更强增强
            base_aug.append(A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5))
        return A.Compose(base_aug)

    def _generate_augmented_samples(self, samples: List[str], label: int,
                                    need_count: int, augmentor: A.Compose) -> List[Tuple[np.ndarray, int]]:
        augmented = []
        for _ in range(need_count):
            base_img = random.choice(samples)
            try:
                img_path = os.path.join(self.root, base_img)
                image = np.array(Image.open(img_path).convert('RGB'))
                augmented_img = augmentor(image=image)['image']
                augmented.append((augmented_img, label))
            except Exception as e:
                print(f"Augmentation failed for {base_img}: {e}")
                continue
        return augmented

    def _print_class_distribution(self, description: str):
        labels = [label for _, label in self.data_dir]
        print(f"\n===== {description} Class Distribution =====")
        for cls, count in sorted(Counter(labels).items()):
            print(f"Class {cls}: {count} samples")
        print("=" * 40)

    def __getitem__(self, idx: int) -> Tuple[jt.Var, Union[int, str]]:
        try:
            img_name, label = self.data_dir[idx]
            if isinstance(img_name, str):
                img_path = os.path.join(self.root, img_name)
                image = Image.open(img_path).convert('RGB')
                image = np.array(image)
            else:
                image = img_name

            # 应用增强
            if self.is_train and self.alb_transform:
                image = Image.fromarray(self.alb_transform(image=np.array(image))['image'])

            if self.hard_augment and label in [2, 3, 4]:
                image = self.hard_augmentor(image, label)

            if self.transform:
                image = self.transform(image)

            return jt.array(image), label if label is not None else img_name.split('.')[0]

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return jt.zeros((3, 224, 224)), -1

# ============== Loss_function ================
class DynamicFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if not isinstance(alpha, jt.Var):
                alpha = jt.array(alpha, dtype=jt.float32)
            elif not alpha.dtype.is_float():
                alpha = alpha.float()
        self.alpha = alpha

    def execute(self, pred, target):
        ce_loss = nn.cross_entropy_loss(pred, target, reduction='none')
        p_t = jt.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        if self.alpha is not None:
            if not target.dtype.is_int():
                target = target.int()
            focal_loss = self.alpha[target] * focal_loss
        return focal_loss.mean()

# ============== Training ==============
def training(model:nn.Module, optimizer:nn.Optimizer, train_loader:Dataset,
             now_epoch:int, num_epochs:int, dw:DynamicWeighting, hard_bank:HardSampleBank):
    model.train()
    losses = []
    all_preds, all_labels = [], []

    gamma = min(2.0, 0.5 + now_epoch * 0.05)  # 逐渐增加对难样本的关注
    criterion = DynamicFocalLoss(gamma=gamma, alpha=dw.class_weights) # 动态调整Focal Loss参数

    pbar = tqdm(train_loader, total=len(train_loader))
    for data in pbar:
        image, label = data

        # 从难样本库中抽取样本混合训练
        if now_epoch > 10 and hard_bank:  # 前10epoch不用难样本
            hard_batch = hard_bank.get_batch(image.shape[0]//2)
            if hard_batch:
                image = jt.concat([image, hard_batch[0]])
                label = jt.concat([label, hard_batch[1]])

        pred = model(image)
        pred.sync()

        loss = criterion(pred, label)
        loss.sync()

        all_preds.append(pred.numpy().argmax(axis=1))
        all_labels.append(label.numpy())

        optimizer.step(loss)
        losses.append(loss.item())
        pbar.set_description(f'Epoch {now_epoch} [TRAIN] loss={losses[-1]:.3f}')

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_labels)
    acc = np.mean(np.float32(preds == targets))

    # 计算每个类别的召回率、精确率、F1-score
    print("\n===== Classification Report =====")
    print(classification_report(
        targets, preds,
        target_names=[f"Class {i}" for i in range(6)],
        digits=4
    ))

    # 打印混淆矩阵
    print("\n===== Confusion Matrix =====")
    print(confusion_matrix(targets, preds))

    # 单独计算难样本准确率
    hard_mask = np.isin(targets, [2,3,4])
    hard_acc = np.mean(preds[hard_mask] == targets[hard_mask]) if np.any(hard_mask) else 0

    print(f'\nEpoch {now_epoch} Train | Loss: {np.mean(losses):.4f} | Acc: {acc:.4f} | Hard Acc: {hard_acc:.4f}')
    return np.mean(losses), acc

def evaluate(model:nn.Module, val_loader:Dataset, hard_bank:HardSampleBank=None):
    model.eval()
    preds, targets, probs = [], [], []
    all_images = []

    for data in val_loader:
        image, label = data
        with jt.no_grad():
            output = model(image)
            prob = nn.softmax(output, dim=1)

        preds.append(output.numpy().argmax(axis=1))
        targets.append(label.numpy())
        probs.append(prob.numpy())
        if hard_bank is not None:
            all_images.append(image.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    probs = np.concatenate(probs)
    acc = np.mean(np.float32(preds == targets))

    # 收集难样本
    if hard_bank is not None:
        all_images = np.concatenate(all_images)
        hard_bank.add(all_images, targets, preds, probs)

    # 计算难样本指标
    hard_mask = np.isin(targets, [2,3,4])
    hard_acc = np.mean(preds[hard_mask] == targets[hard_mask]) if np.any(hard_mask) else 0

    print(f'\n===== Validation =====')
    print(f'Overall Acc: {acc:.4f} | Hard Classes Acc: {hard_acc:.4f}')
    print(classification_report(targets, preds, digits=4))
    print(confusion_matrix(targets, preds))

    return acc, confusion_matrix(targets, preds, labels=range(6))

def run(model:nn.Module, optimizer:nn.Optimizer, train_loader:Dataset, val_loader:Dataset, num_epochs:int, modelroot:str):
    best_acc = 0
    history = []
    dw = DynamicWeighting(num_classes=6)
    hard_bank = HardSampleBank()

    # 学习率调度
    warmup_epochs = 10
    base_lr = 5e-5

    for epoch in range(num_epochs):
        # 学习率调整
        if epoch < warmup_epochs:  # 学习率预热
            lr = base_lr * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else: #余弦退火
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs-warmup_epochs, eta_min=1e-6)
            scheduler.step()

        train_loss, train_acc = training(model, optimizer, train_loader, epoch, num_epochs, dw, hard_bank)
        val_acc, val_conf = evaluate(model, val_loader, hard_bank)

        # 更新类别权重
        dw.update(val_conf)

        # 保存记录
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })

        # 保存模型
        if val_acc > best_acc:
            best_acc = val_acc
            model.save(os.path.join(modelroot, 'best_model.pkl'))

        if epoch > 20:
            if epoch % 10 == 0:
                model.save(os.path.join(modelroot, f'epoch_{epoch}.pkl'))

        print(f'Epoch {epoch} Summary | Best Val Acc: {best_acc:.4f} | Current: {val_acc:.4f}')

    # 保存训练历史
    pd.DataFrame(history).to_csv(os.path.join(modelroot, 'training_history.csv'), index=False)

# ============== Test ==================
def test(model: nn.Module, test_loader: Dataset, result_path: str):
    model.eval()
    preds = []
    filenames = []

    print("Testing...")
    for data in test_loader:
        images, names = data
        with jt.no_grad():
            outputs = model(images)
        preds.append(outputs.numpy().argmax(axis=1))
        filenames.extend(names)

    # 强制添加 .jpg 后缀（如果文件名没有后缀）
    filenames = [f"{name}.jpg" if not name.endswith('.jpg') else name for name in filenames]

    # 自然排序（按文件名中的数字部分）
    def natural_sort_key(s):
        base_name = s.split('.')[0]  # 去掉 .jpg 后按数字排序
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', base_name)]

    sorted_indices = sorted(range(len(filenames)), key=lambda i: natural_sort_key(filenames[i]))
    filenames_sorted = [filenames[i] for i in sorted_indices]  # 带 .jpg
    preds_sorted = np.concatenate(preds)[sorted_indices]

    with open(result_path, 'w') as f:
        for name, pred in zip(filenames_sorted, preds_sorted):
            f.write(f"{name} {pred}\n")

# ============== Main ==============
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) #定位到Jittor_Env的位置
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default=os.path.join(project_root, 'code/JT/baseline/TrainSet'))
    parser.add_argument('--modelroot', type=str, default=os.path.join(project_root, 'code/JT/baseline/model_save_2'))
    parser.add_argument('--testonly', action='store_true', default=False)
    parser.add_argument('--loadfrom', type=str, default=os.path.join(project_root, 'checkpoints/checkpoint2.pkl'))
    parser.add_argument('--result_path', type=str, default=os.path.join(project_root, 'code/JT/baseline/result_2.txt'))
    args = parser.parse_args()

    # 在主函数中添加模型冻结逻辑
    if not args.testonly:
        model = swin_base_patch4_window7_224_in22k(pretrained=True, num_classes=6)

        # 冻结底层参数
        for name, param in model.named_parameters():
            if 'head' not in name:  # 只训练最后的分类头
                param.requires_grad = False

        # 解冻最后3个阶段
        for name, param in model.named_parameters():
            if 'layers.3' in name or 'layers.2' in name:
                param.requires_grad = True

    # 修改transform加入难样本增强
    transform_train = Compose([
        Resize((224, 224)),
        RandomCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    #修改transform加入难样本增强
    transform_val = Compose([
        Resize((224, 224)),
        RandomCrop(224),
        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if not args.testonly:
        optimizer = nn.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

        train_loader = ImageFolder(
            root=os.path.join(args.dataroot, 'images/train'),
            annotation_path=os.path.join(args.dataroot, 'labels/train.txt'),
            transform=transform_train,
            is_train=True,
            oversample_minority=True,
            hard_augment=True,  # 启用难样本增强
            batch_size=8,
            num_workers=8,
            shuffle=True
        )

        val_loader = ImageFolder(
            root=os.path.join(args.dataroot, 'images/train'),
            annotation_path=os.path.join(args.dataroot, 'labels/val.txt'),
            transform=transform_val,
            oversample_minority=True,
            is_train=False,
            batch_size=8,
            num_workers=8,
            shuffle=False
        )

        run(model, optimizer, train_loader, val_loader, 200, args.modelroot)
    else:
        test_loader = ImageFolder(
            root=os.path.join(project_root, "code/JT/baseline/TestSetA"),
            transform=transform_val,
            oversample_minority=False,
            is_train=False,
            batch_size=8,
            num_workers=8,
            shuffle=False
        )
        model = swin_base_patch4_window7_224_in22k(pretrained=True, num_classes=6)
        model.load(args.loadfrom)
        test(model, test_loader, args.result_path)