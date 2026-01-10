import json
import os
from typing import Dict, Optional

from astrbot.api import logger


class EmotionManager:
    """处理所有与感情数据相关的加载、保存和管理逻辑"""

    def __init__(self, file_path):
        """
        初始化感情管理器。
        :param file_path: emotions.json 文件的路径。
        """
        self.file_path = file_path
        self.emotions_data: Dict = self._load_emotions_from_file()

    def _load_emotions_from_file(self) -> Dict:
        """从JSON文件加载感情数据"""
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(
                    f"成功从 {self.file_path} 加载 {sum(len(v) for v in data.values())} 个感情配置。"
                )
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"加载感情文件失败: {e}")
            return {}

    def _save_emotions_to_file(self) -> bool:
        """将当前感情数据保存到JSON文件"""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.emotions_data, f, ensure_ascii=False, indent=4)
            return True
        except IOError as e:
            logger.error(f"保存感情文件失败: {e}")
            return False

    def reload(self):
        """从文件重新加载数据，用于在保存失败时恢复状态"""
        self.emotions_data = self._load_emotions_from_file()

    def get_emotion_data(
        self, character_name: str, emotion_name: str
    ) -> Optional[Dict]:
        """获取指定角色和感情的数据"""
        return self.emotions_data.get(character_name, {}).get(emotion_name)

    def character_exists(self, character_name: str) -> bool:
        """检查角色是否存在"""
        return character_name in self.emotions_data

    def register_emotion(
        self,
        character_name: str,
        emotion_name: str,
        ref_audio_path: str,
        ref_audio_text: str,
        language: str = None,
    ) -> bool:
        """注册一个新的感情并保存"""
        if character_name not in self.emotions_data:
            self.emotions_data[character_name] = {}

        data = {
            "ref_audio_path": ref_audio_path,
            "ref_audio_text": ref_audio_text,
        }
        if language:
            data["language"] = language

        self.emotions_data[character_name][emotion_name] = data
        return self._save_emotions_to_file()

    def delete_emotion(self, character_name: str, emotion_name: str) -> bool:
        """删除一个已注册的感情并保存"""
        if not self.get_emotion_data(character_name, emotion_name):
            return True  # 如果不存在，也视为成功

        del self.emotions_data[character_name][emotion_name]
        if not self.emotions_data[character_name]:
            del self.emotions_data[character_name]

        return self._save_emotions_to_file()
