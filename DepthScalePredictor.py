class DepthScalePredictor(nn.Module):
    """深度尺度预测模块"""
    def __init__(self, in_channels=3, hidden_dim=128, num_layers=3):
        super().__init__()
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        )
        
        # 尺度回归网络
        scale_layers = []
        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else 256
            out_dim = hidden_dim if i < num_layers-1 else 1
            scale_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers-1:
                scale_layers.append(nn.ReLU())
        self.scale_regressor = nn.Sequential(*scale_layers)
        
        # 偏移量回归网络
        offset_layers = []
        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else 256
            out_dim = hidden_dim if i < num_layers-1 else 1
            offset_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers-1:
                offset_layers.append(nn.ReLU())
        self.offset_regressor = nn.Sequential(*offset_layers)
        
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        # 初始化尺度回归为1，偏移为0
        nn.init.constant_(self.scale_regressor[-1].weight, 0)
        nn.init.constant_(self.scale_regressor[-1].bias, 1.0)
        nn.init.constant_(self.offset_regressor[-1].weight, 0)
        nn.init.constant_(self.offset_regressor[-1].bias, 0.0)

    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # 展平
        
        # 预测尺度因子和偏移量
        scale = self.scale_regressor(features)
        offset = self.offset_regressor(features)
        
        # 应用激活函数确保尺度为正
        scale = F.softplus(scale) + 1e-6  # 确保尺度大于0
        
        return scale, offset


class PoseNetWithScaleCorrection(nn.Module):
    """带深度尺度校正的位姿估计网络"""
    def __init__(self, base_model_cfg, depth_model_path):
        super().__init__()
        # 基础位姿估计模型
        self.base_model = build_model(base_model_cfg)
        
        # 深度估计模型 (DepthAnythingV2)
        self.depth_model = self._init_depth_model(depth_model_path)
        
        # 深度尺度预测模块
        self.scale_predictor = DepthScalePredictor(in_channels=3)
        
        # 融合模块 (将深度信息融入位姿估计)
        self.fusion_block = nn.Sequential(
            nn.Conv2d(256 + 1, 256, 3, padding=1),  # 假设基础模型特征为256通道
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        
        # 初始化融合层的权重
        self._init_fusion_weights()

    def _init_depth_model(self, model_path):
        """初始化DepthAnything模型"""
        model = DPT_DINOv2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _init_fusion_weights(self):
        """初始化融合层权重"""
        nn.init.kaiming_normal_(self.fusion_block[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fusion_block[0].bias, 0)
        nn.init.kaiming_normal_(self.fusion_block[2].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fusion_block[2].bias, 0)

    def forward(self, real_images, rendered_depths=None):
        # 1. 深度估计
        with torch.no_grad():
            pseudo_depth = self.depth_model(real_images)
        
        # 2. 深度尺度校正
        scale, offset = self.scale_predictor(real_images)
        corrected_depth = scale.view(-1, 1, 1, 1) * pseudo_depth + offset.view(-1, 1, 1, 1)
        
        # 3. 基础特征提取
        base_features = self.base_model.extract_features(real_images)
        
        # 4. 深度特征融合
        depth_features = self.fusion_block(torch.cat([base_features, corrected_depth], dim=1))
        
        # 5. 位姿估计
        pose_output = self.base_model.pose_head(depth_features)
        
        # 6. 返回结果 (包括校正后的深度用于损失计算)
        return {
            'pose': pose_output,
            'corrected_depth': corrected_depth,
            'scale': scale,
            'offset': offset
        }