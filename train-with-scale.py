def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_pose_loss = 0
    total_depth_loss = 0
    
    for batch_idx, data_batch in enumerate(data_loader):
        # 格式化数据
        formatted_data = format_data_train_sup(data_batch)
        real_images = formatted_data['real_images'].to(device)
        rendered_depths = formatted_data['rendered_depths'].to(device)
        
        # 前向传播
        outputs = model(real_images)
        
        # 计算位姿损失
        gt_rotations = formatted_data['gt_rotations'].to(device)
        gt_translations = formatted_data['gt_translations'].to(device)
        pose_loss = pose_loss_function(
            outputs['pose'], 
            (gt_rotations, gt_translations)
        )
        
        # 计算深度损失 (使用校正后的深度)
        corrected_depth = outputs['corrected_depth']
        depth_loss = F.l1_loss(corrected_depth, rendered_depths)
        
        # 组合损失
        total_loss = pose_loss + 0.5 * depth_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 记录损失
        total_pose_loss += pose_loss.item()
        total_depth_loss += depth_loss.item()
    
    return total_pose_loss / len(data_loader), total_depth_loss / len(data_loader)
	
class ScaleConsistencyLoss(nn.Module):
    """深度尺度一致性损失"""
    def __init__(self, target_scale=1.0, weight=0.1):
        super().__init__()
        self.target_scale = target_scale
        self.weight = weight
        
    def forward(self, scale):
        # 鼓励尺度接近目标值 (例如1.0)
        scale_loss = F.mse_loss(scale, torch.ones_like(scale) * self.target_scale)
        return self.weight * scale_loss

# 在训练中使用
scale_loss = scale_consistency_loss(outputs['scale'])
total_loss = pose_loss + 0.5 * depth_loss + scale_loss	
