import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, ignore_bg=False) -> None:
        super().__init__()
        self.nonline = nn.Softmax(dim=1)
        self.smooth = 1e-5
        self.ignore_bg = ignore_bg
    
    def forward(self, x, y):
        with torch.no_grad():
            y_onehot = torch.zeros_like(x)
            y_onehot = y_onehot.scatter(1, y.long(), 1)
        axes = [0] + list(range(2, len(x.shape)))
        
        x = self.nonline(x)

        tp = (x * y_onehot).sum(axes)
        fp = (x * (1 - y_onehot)).sum(axes)
        fn = ((1 - x) * y_onehot).sum(axes)
        
        numerator = 2. * tp + self.smooth
        denominator = 2. * tp + fp + fn + self.smooth
        dc = numerator / (denominator + 1e-8)
        dc = dc[1:].mean() if self.ignore_bg else dc.mean()
        return -dc

class DiceCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dc = DiceLoss(ignore_bg=True)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, y):
        dc_loss = self.dc(x, y)
        ce_loss = self.ce(x, y.squeeze(1).long())
        return dc_loss + ce_loss

class CELoss_p(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, y):
        ce_loss = self.ce(x, y.long())
        return ce_loss

class CELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weight = torch.tensor([0.1, 0.2, 0.3, 0.5]).to(torch.device("cuda:0"))
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, x, y):
        ce_loss = self.ce(x, y.squeeze(1).long())
        return ce_loss

class MultiOutLoss(nn.Module):
    """
    wrap any loss function for the use of deep supervision
    """
    def __init__(self, loss_function, weights) -> None:
        super().__init__()
        self.loss_function = loss_function
        self.weights = weights

    def forward(self, x, y):
        l = self.weights[0] * self.loss_function(x[0], y[0])
        for i in range(1, len(x)):
            l += self.weights[i] * self.loss_function(x[i], y[i])
        return l

# 对比损失（NT-Xent）
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_matrix = sim_matrix / self.temperature
        labels = torch.arange(batch_size).to(z_i.device)
        labels = torch.cat([labels, labels], dim=0)
        loss = self.criterion(sim_matrix, labels)
        return loss / (2 * batch_size)


class RegionContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.6):
        super(RegionContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feat1, feat2, mask):
        """
        feat1, feat2: Tensor [B, C, H, W] - 两个不同增强视图下的特征
        mask: Tensor [B, H, W] - 伪标签区域（每个位置是类别或 bool）
        """
        B, C, H, W = feat1.shape
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)

        total_loss = 0.0
        valid_batch = 0

        for b in range(B):
            pos_mask = mask[b] > 0  # 假设mask中0是背景或无效区域
            if pos_mask.sum() == 0:
                continue

            v1 = feat1[b, :, pos_mask].T  # [N, C]
            v2 = feat2[b, :, pos_mask].T  # [N, C]

            sim_matrix = torch.matmul(v1, v2.T) / self.temperature  # [N, N]
            labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            total_loss += F.cross_entropy(sim_matrix, labels)
            valid_batch += 1

        if valid_batch == 0:
            return torch.tensor(0.0, device=feat1.device, requires_grad=True)
        else:
            return total_loss / valid_batch


class SoftmaxMSELoss(nn.Module):
    """
    返回形状为 (B,) 的逐样本 MSE，不再 .sum()；
    如需平均，后续再除以 B 即可。
    """
    def __init__(self, sigmoid: bool = False):
        super().__init__()
        self.sigmoid = sigmoid

    def forward(self, input_logits, target_logits):
        assert input_logits.shape == target_logits.shape
        if self.sigmoid:
            input_prob  = torch.sigmoid(input_logits)
            target_prob = torch.sigmoid(target_logits)
        else:
            input_prob  = F.softmax(input_logits,  dim=1)
            target_prob = F.softmax(target_logits, dim=1)

        # 将每个样本展平后计算 MSE，保留 batch 维度
        return ((input_prob - target_prob).flatten(1) ** 2).mean()


class KLDistillLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.T = temperature
        self.kl = nn.KLDivLoss(reduction="batchmean")  # 已按 batch 平均

    def forward(self, input_logits, target_logits):
        assert input_logits.shape == target_logits.shape
        assert not torch.isnan(input_logits).any(), "input_logits has NaN"
        assert not torch.isnan(target_logits).any(), "target_logits has NaN"
        student_logp = F.log_softmax(input_logits / self.T, dim=1)
        teacher_p = F.softmax(target_logits / self.T, dim=1).clamp(min=1e-8)
        return self.kl(student_logp, teacher_p) * (self.T ** 2)

class JSDConsistencyLoss(nn.Module):
    def __init__(self):
        super(JSDConsistencyLoss, self).__init__()

    def forward(self, pred, target_soft, conf_weights):
        pred_soft = F.softmax(pred, dim=1)
        M = 0.5 * (pred_soft + target_soft)
        kl1 = F.kl_div(F.log_softmax(pred, dim=1), M, reduction='none')
        kl2 = F.kl_div(torch.log(target_soft + 1e-6), M, reduction='none')
        jsd_loss = 0.5 * (kl1 + kl2)
        weighted_jsd_loss = jsd_loss * conf_weights.unsqueeze(1)
        return weighted_jsd_loss.sum() / (conf_weights.sum() + 1e-6)


import torch
import torch.nn as nn

def sigmoid_rampup(current, rampup_length, min_threshold=0.3, max_threshold=0.5):
    if rampup_length == 0:
        return max_threshold
    current = float(current)
    phase = 1.0 - max(0.0, min(1.0, current / rampup_length))
    return min_threshold + (max_threshold - min_threshold) * (1.0 - phase)

class FeCLoss(nn.Module):
    def __init__(self, temperature=0.6, gamma=2.0, use_focal=False, rampup_epochs=100, lambda_cross=1.0):
        super(FeCLoss, self).__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.use_focal = use_focal
        self.rampup_epochs = rampup_epochs
        self.lambda_cross = lambda_cross

    def forward(self, feat, mask, teacher_feat=None, gambling_uncertainty=None, epoch=0):
        """
        feat: (B, N, D)
        mask: (B, 1, N)
        teacher_feat: (B, N, D)
        gambling_uncertainty: (B, N)
        """
        feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=True)
        B, C, H, W = feat.shape
        N = H * W
        mask = F.avg_pool2d(mask, kernel_size=8, stride=8)
        mask = mask.squeeze(1).view(B, -1).unsqueeze(1)
        feat = feat.view(B, C, N)
        feat = torch.transpose(feat, 1, 2)
        feat = F.normalize(feat, dim=-1)
        device = feat.device
        if teacher_feat is not None:
            teacher_feat = F.interpolate(teacher_feat, scale_factor=2, mode='bilinear', align_corners=True)
            teacher_feat = teacher_feat.view(B, C, N)
            teacher_feat = torch.transpose(teacher_feat, 1, 2)
            teacher_feat = F.normalize(teacher_feat, dim=-1)
        # Positive/Negative masks
        mem_mask = torch.eq(mask, mask.transpose(1, 2)).float()  # (B, N, N)
        mem_mask_neg = 1 - mem_mask

        feat_logits = torch.matmul(feat, feat.transpose(1, 2)) / self.temperature  # (B, N, N)
        identity = torch.eye(N, device=device).unsqueeze(0)  # (1, N, N)
        neg_identity = 1 - identity  # mask out self-similarity
        feat_logits = feat_logits * neg_identity

        feat_logits_max, _ = torch.max(feat_logits, dim=1, keepdim=True)
        feat_logits = feat_logits - feat_logits_max.detach()
        exp_logits = torch.exp(feat_logits)

        neg_sum = torch.sum(exp_logits * mem_mask_neg, dim=-1)
        denominator = exp_logits + neg_sum.unsqueeze(-1)
        division = exp_logits / (denominator + 1e-18)

        loss_matrix = -torch.log(division + 1e-18)
        loss_matrix = loss_matrix * mem_mask * neg_identity  # exclude self

        loss_student = torch.sum(loss_matrix, dim=-1) / (torch.sum(mem_mask, dim=-1) - 1 + 1e-18)
        loss_student = loss_student.mean()

        # Focal weighting
        if self.use_focal:
            similarity = division
            focal_weights = torch.ones_like(similarity)
            pos_thresh = sigmoid_rampup(epoch, self.rampup_epochs, min_threshold=1.3, max_threshold=1.5)
            neg_thresh = sigmoid_rampup(epoch, self.rampup_epochs, min_threshold=0.3, max_threshold=0.5)

            hard_pos_mask = mem_mask.bool() & (similarity < pos_thresh)
            hard_neg_mask = mem_mask_neg.bool() & (similarity > neg_thresh)

            focal_weights[hard_pos_mask] = (1 - similarity[hard_pos_mask]).pow(self.gamma)
            focal_weights[hard_neg_mask] = similarity[hard_neg_mask].pow(self.gamma)

            loss_student = torch.sum(loss_matrix * focal_weights, dim=-1) / (torch.sum(mem_mask, dim=-1) - 1 + 1e-18)
            loss_student = loss_student.mean()

        # Gambling uncertainty modulation
        if gambling_uncertainty is not None:
            loss_student_per_patch = torch.sum(loss_matrix, dim=-1) / (torch.sum(mem_mask, dim=-1) - 1 + 1e-18)
            loss_student = (loss_student_per_patch * gambling_uncertainty).mean()

        # Auxiliary teacher-student contrastive loss
        loss_cross = 0.0
        if teacher_feat is not None:
            cross_sim = torch.matmul(feat, teacher_feat.transpose(1, 2))
            cross_sim.clamp(max=1 - 1e-6)
            mem_mask_cross = torch.eq(mask, mask.transpose(1, 2)).float()
            mem_mask_cross_neg = 1 - mem_mask_cross

            cross_neg_thresh = sigmoid_rampup(epoch, self.rampup_epochs, min_threshold=0.3, max_threshold=0.5)
            cross_hard_neg_mask = mem_mask_cross_neg.bool() & (cross_sim > cross_neg_thresh)

            if cross_hard_neg_mask.sum() > 0:
                loss_cross_term = -torch.log(1 - cross_sim + 1e-6)
                loss_cross_term = loss_cross_term * cross_hard_neg_mask.float()
                loss_cross = torch.sum(loss_cross_term) / (torch.sum(cross_hard_neg_mask.float()) + 1e-18)

        total_loss = loss_student + self.lambda_cross * loss_cross
        return total_loss