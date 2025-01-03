import json
import os
from typing import Dict, Any, List, Union

class ConfigLoader:
    """配置加载器类，用于管理所有配置项"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置加载器
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        # 获取配置文件所在目录的路径
        self.config_dir = os.path.dirname(os.path.abspath(config_path))
        # 获取项目根目录的路径（landscape_trace目录）
        self.project_root = os.path.dirname(self.config_dir)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = json.load(f)
            
        # 验证配置文件版本
        self._version = self._config.get('version', '1.0.0')
        
    @property
    def version(self) -> str:
        """获取配置文件版本"""
        return self._version
        
    def get_model_path(self, model_name: str) -> str:
        """获取模型文件路径"""
        # 处理特殊情况：PZ+DL -> pz_dl
        model_name = model_name.lower().replace('+', '_')
        relative_path = self._config['model']['paths'].get(model_name)
        if relative_path:
            # 将相对路径转换为绝对路径
            return os.path.join(self.project_root, relative_path)
        return None
        
    def get_model_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return self._config['model']['params']
        
    def get_process_order(self) -> List[str]:
        """获取处理顺序"""
        return self._config['model']['process_order']
        
    def get_image_params(self) -> Dict[str, int]:
        """获取图像处理参数"""
        return self._config['image']['params']
        
    def get_color_map(self) -> Dict[str, List[int]]:
        """获取颜色映射"""
        return self._config['colors']
        
    def get_process_params(self, element_type: str = 'default') -> Dict[str, Union[int, float]]:
        """
        获取处理参数
        Args:
            element_type: 元素类型，默认使用default配置
        """
        params = self._config['process']['params']
        return params.get(element_type, params['default'])
        
    def get_all_model_paths(self) -> List[str]:
        """获取所有模型路径"""
        paths = self._config['model']['paths'].values()
        return [os.path.join(self.project_root, path) for path in paths]
        
    def __getitem__(self, key: str) -> Any:
        """允许直接访问配置项"""
        return self._config[key]
        
    def validate_paths(self) -> bool:
        """验证所有模型文件路径是否存在"""
        # print("\n检查模型文件路径：")
        # print(f"项目根目录: {self.project_root}")
        all_valid = True
        for path in self.get_all_model_paths():
            exists = os.path.exists(path)
            # print(f"模型文件: {path} ({'存在' if exists else '不存在'})")
            if not exists:
                all_valid = False
        return all_valid

# 创建全局配置实例
config = ConfigLoader() 