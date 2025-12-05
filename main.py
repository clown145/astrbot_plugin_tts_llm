import asyncio
import httpx
import os
import re
from typing import Dict, Optional, Set

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger, AstrBotConfig
import astrbot.api.message_components as Comp
from astrbot.api.provider import LLMResponse

# 从新模块导入功能
from .emotion_manager import EmotionManager
from .tts_engine import TTSEngine
from .external_apis import translate_text


@register(
    "astrbot_plugin_tts_llm",
    "clown145",
    "一个通过LLM、翻译和TTS实现语音合成的插件",
    "1.2.1",
    "https://github.com/clown145/astrbot_plugin_tts_llm",
)
class LlmTtsPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.active_sessions: Set[str] = set()
        self.w_active_sessions: Set[str] = set()
        self.session_emotions: Dict[str, Dict[str, str]] = {}
        self.session_w_settings: Dict[str, Dict[str, str]] = {}
        self._keepalive_stop_event = asyncio.Event()
        self._keepalive_task: Optional[asyncio.Task] = None

        # 初始化辅助模块
        plugin_data_dir = StarTools.get_data_dir("astrbot_plugin_tts_llm")
        emotions_file_path = plugin_data_dir / "emotions.json"
        self.emotion_manager = EmotionManager(emotions_file_path)
        
        self.http_client = httpx.AsyncClient(timeout=300.0)
        self.tts_engine = TTSEngine(self.config, self.http_client)

        if self.config.get("enable_space_keepalive"):
            self._keepalive_task = asyncio.create_task(self._keep_alive_loop())

        logger.info("LLM TTS 插件已加载。")

    def _get_keepalive_url(self) -> Optional[str]:
        """获取用于保活的目标地址，优先使用独立配置，否则回落到第一个TTS服务器。"""

        custom_url = self.config.get("space_keepalive_url")
        if custom_url:
            return custom_url.rstrip("/")

        servers = self.config.get("tts_servers", [])
        if servers:
            return servers[0].rstrip("/")

        return None

    async def _keep_alive_loop(self):
        """定时ping Hugging Face Space 以避免因长时间无人访问而休眠。"""

        interval_minutes = max(int(self.config.get("space_keepalive_interval_minutes", 25)), 1)

        while not self._keepalive_stop_event.is_set():
            target_url = self._get_keepalive_url()
            if not target_url:
                logger.warning("未找到可用于保活的空间地址，已跳过保活任务。")
                await asyncio.wait_for(self._keepalive_stop_event.wait(), timeout=interval_minutes * 60)
                continue

            try:
                response = await self.http_client.get(target_url, timeout=30)
                logger.info(f"保活请求已发送到 {target_url}，状态码: {response.status_code}")
            except Exception as exc:
                logger.warning(f"向 {target_url} 发送保活请求失败: {exc}")

            try:
                await asyncio.wait_for(self._keepalive_stop_event.wait(), timeout=interval_minutes * 60)
            except asyncio.TimeoutError:
                continue

    @filter.command("注册感情")
    async def register_emotion_command(
        self, event: AstrMessageEvent, character_name: str, emotion_name: str, ref_audio_path: str, ref_audio_text: str,
    ):
        """注册一个新的感情并保存到文件"""
        if ".." in ref_audio_path or os.path.isabs(ref_audio_path):
            yield event.plain_result("❌ 错误：参考音频路径无效。它必须是一个相对路径，且不能包含 '..'。" )
            return

        if self.emotion_manager.register_emotion(character_name, emotion_name, ref_audio_path, ref_audio_text):
            yield event.plain_result(f"✅ 感情 '{emotion_name}' 已成功注册到角色 '{character_name}' 下。")
        else:
            self.emotion_manager.reload()  # 如果保存失败，从文件重新加载以恢复状态
            yield event.plain_result("❌ 保存感情时发生错误，注册失败。")

    @filter.command("删除感情")
    async def delete_emotion_command(self, event: AstrMessageEvent, character_name: str, emotion_name: str):
        """删除一个已注册的感情"""
        if not self.emotion_manager.character_exists(character_name):
            yield event.plain_result(f"❌ 错误：未找到角色 '{character_name}'。")
            return

        if not self.emotion_manager.get_emotion_data(character_name, emotion_name):
            yield event.plain_result(f"❌ 错误：角色 '{character_name}' 下未找到名为 '{emotion_name}' 的感情。")
            return

        if self.emotion_manager.delete_emotion(character_name, emotion_name):
            yield event.plain_result(f"✅ 已成功删除角色 '{character_name}' 的感情 '{emotion_name}'。")
        else:
            self.emotion_manager.reload() # 如果保存失败，从文件重新加载以恢复状态
            yield event.plain_result("❌ 保存文件时发生错误，删除失败。")

    @filter.command("查看感情")
    async def view_emotions_command(self, event: AstrMessageEvent):
        """查看所有已注册的感情"""
        emotions_data = self.emotion_manager.emotions_data
        if not emotions_data:
            yield event.plain_result("当前未注册任何感情。")
            return

        formatted_lines = ["所有已注册的感情列表："]
        for character, emotions in emotions_data.items():
            formatted_lines.append(f"\n角色: {character}")
            if emotions:
                for emotion_name in emotions.keys():
                    formatted_lines.append(f"  - {emotion_name}")
            else:
                formatted_lines.append("  (暂无感情)")

        final_message = "\n".join(formatted_lines)
        yield event.plain_result(final_message)

    @filter.command("合成")
    async def direct_tts_command(self, event: AstrMessageEvent, character_name: str, emotion_name: str, text_to_synthesize: str):
        """根据角色和感情名直接合成语音"""
        emotion_data = self.emotion_manager.get_emotion_data(character_name, emotion_name)
        if not emotion_data:
            yield event.plain_result(f"❌ 未找到角色 '{character_name}' 的感情 '{emotion_name}'。请先使用 /注册感情 指令添加。")
            return

        yield event.plain_result("收到合成请求，正在处理...")
        audio_path = await self.tts_engine.synthesize(
            character_name=character_name,
            ref_audio_path=emotion_data["ref_audio_path"],
            ref_audio_text=emotion_data["ref_audio_text"],
            text=text_to_synthesize,
            session_id_for_log=event.unified_msg_origin,
        )

        if audio_path:
            yield event.chain_result([Comp.Record(file=audio_path)])
        else:
            yield event.plain_result("语音合成失败，请检查服务器状态或日志。")
        event.stop_event()

    @filter.command("tts-llm", alias={"开启语音合成"})
    async def start_tts(self, event: AstrMessageEvent):
        session_id = event.unified_msg_origin
        self.active_sessions.add(session_id)
        self.w_active_sessions.discard(session_id)
        default_char = self.config.get("default_character")
        default_emotion = self.config.get("default_emotion_name")
        logger.info(f"会话 [{session_id}] 的 LLM TTS 功能已开启。")
        yield event.plain_result(f"▶️ 本对话的LLM语音合成已开启。\n将使用默认感情: {default_char} - {default_emotion}")

    @filter.command("tts-q", alias={"关闭语音合成"})
    async def stop_tts(self, event: AstrMessageEvent):
        session_id = event.unified_msg_origin
        self.active_sessions.discard(session_id)
        self.w_active_sessions.discard(session_id)
        logger.info(f"会话 [{session_id}] 的所有 LLM TTS 功能已关闭。")
        yield event.plain_result("⏹️ 本对话的所有LLM语音合成功能已关闭。")

    @filter.command("tts-w", alias={"开启自动情感识别"})
    async def start_tts_w(self, event: AstrMessageEvent):
        session_id = event.unified_msg_origin
        self.w_active_sessions.add(session_id)
        self.active_sessions.discard(session_id)
        default_char = self.config.get("default_character")
        logger.info(f"会话 [{session_id}] 的 LLM 自动情感识别 TTS 功能已开启。")
        yield event.plain_result(f"▶️ 本对话的自动情感识别语音合成已开启。\n将使用默认角色: {default_char}")
    
    @filter.command("tts-w-q", alias={"关闭自动情感识别"})
    async def stop_tts_w(self, event: AstrMessageEvent):
        session_id = event.unified_msg_origin
        self.w_active_sessions.discard(session_id)
        logger.info(f"会话 [{session_id}] 的 LLM 自动情感识别 TTS 功能已关闭。")
        yield event.plain_result("⏹️ 本对话的自动情感识别语音合成已关闭。")

    @filter.command("sw", alias={"切换感情"})
    async def switch_emotion(self, event: AstrMessageEvent, character_name: str, emotion_name: str):
        if self.emotion_manager.get_emotion_data(character_name, emotion_name):
            self.session_emotions[event.unified_msg_origin] = {"character": character_name, "emotion": emotion_name}
            logger.info(f"会话 [{event.unified_msg_origin}] 切换感情至: {character_name} - {emotion_name}")
            yield event.plain_result(f"本会话感情已切换为: {character_name} - {emotion_name}")
        else:
            yield event.plain_result(f"❌ 未找到角色 '{character_name}' 的感情 '{emotion_name}'。")

    @filter.command("sw-w", alias={"切换w角色"})
    async def switch_w_character(self, event: AstrMessageEvent, character_name: str):
        if self.emotion_manager.character_exists(character_name):
            self.session_w_settings[event.unified_msg_origin] = {"character": character_name}
            logger.info(f"会话 [{event.unified_msg_origin}] 切换自动情感识别角色至: {character_name}")
            yield event.plain_result(f"本会话自动情感识别角色已切换为: {character_name}")
        else:
            yield event.plain_result(f"❌ 未找到角色 '{character_name}'。")

    async def _synthesize_speech_from_context(self, text: str, session_id: str) -> Optional[str]:
        """根据当前会话设置合成语音（固定感情模式）"""
        session_setting = self.session_emotions.get(session_id)
        char_name, emotion_name = ((session_setting["character"], session_setting["emotion"]) if session_setting 
                                   else (self.config.get("default_character"), self.config.get("default_emotion_name")))
        
        if not char_name or not emotion_name:
            logger.error(f"[{session_id}] 未配置默认角色或感情。")
            return None
        
        emotion_data = self.emotion_manager.get_emotion_data(char_name, emotion_name)
        if not emotion_data:
            logger.error(f"[{session_id}] 找不到感情配置: {char_name} - {emotion_name}")
            return None
            
        return await self.tts_engine.synthesize(
            character_name=char_name,
            ref_audio_path=emotion_data["ref_audio_path"],
            ref_audio_text=emotion_data["ref_audio_text"],
            text=text,
            session_id_for_log=session_id,
        )

    @filter.on_llm_response()
    async def intercept_llm_response_for_tts(self, event: AstrMessageEvent, resp: LLMResponse):
        session_id = event.unified_msg_origin
        original_text = resp.completion_text.strip()
        if not original_text:
            return

        audio_path: Optional[str] = None
        
        if session_id in self.w_active_sessions:
            logger.info(f"[{session_id}] 捕获LLM文本，准备进行自动情感语音合成: {original_text}")
            char_name = self.session_w_settings.get(session_id, {}).get("character") or self.config.get("default_character")

            if not char_name or not self.emotion_manager.character_exists(char_name):
                resp.result_chain.chain.append(Comp.Plain(f"\n(语音合成失败: 角色'{char_name}'未配置或无感情)"))
                return
            
            character_emotions = self.emotion_manager.emotions_data[char_name]
            api_config = self.config.get("translation_api", {})
            w_prompt_template = api_config.get("w_mode_prompt")
            
            if not w_prompt_template:
                resp.result_chain.chain.append(Comp.Plain("\n(语音合成失败: 缺少提示词配置)"))
                return

            emotion_list_str = ", ".join(character_emotions.keys())
            augmented_prompt = w_prompt_template.format(emotion_list=emotion_list_str, text=original_text)
            
            japanese_text_with_emotion = await translate_text(augmented_prompt, self.http_client, api_config, w_prompt_template)
            if not japanese_text_with_emotion:
                resp.result_chain.chain.append(Comp.Plain("\n(翻译或情感识别失败)"))
                return

            match = re.search(r'(.*)\[(.+?)\]\s*$', japanese_text_with_emotion.strip(), re.DOTALL)
            if not match:
                resp.result_chain.chain.append(Comp.Plain("\n(语音合成失败: 无法解析情感)"))
                return

            japanese_text, emotion_name = match.group(1).strip(), match.group(2).strip()
            emotion_data = self.emotion_manager.get_emotion_data(char_name, emotion_name)

            if not emotion_data:
                resp.result_chain.chain.append(Comp.Plain(f"\n(语音合成失败: 情感'{emotion_name}'无效)"))
                return

            audio_path = await self.tts_engine.synthesize(
                character_name=char_name, ref_audio_path=emotion_data["ref_audio_path"],
                ref_audio_text=emotion_data["ref_audio_text"], text=japanese_text, session_id_for_log=session_id
            )

        elif session_id in self.active_sessions:
            logger.info(f"[{session_id}] 捕获LLM文本，准备语音合成: {original_text}")
            api_config = self.config.get("translation_api", {})
            japanese_text = await translate_text(original_text, self.http_client, api_config)
            if not japanese_text:
                resp.result_chain.chain.append(Comp.Plain("\n(翻译失败)"))
                return
            audio_path = await self._synthesize_speech_from_context(japanese_text, session_id)
        
        if audio_path:
            resp.result_chain.chain = [Comp.Record(file=audio_path)]
            if self.config.get("send_text_with_audio", False):
                resp.result_chain.chain.append(Comp.Plain(f"{original_text}"))
        elif session_id in self.active_sessions or session_id in self.w_active_sessions:
            resp.result_chain.chain.append(Comp.Plain("\n(语音合成失败)"))

    async def terminate(self):
        """插件卸载/停用时关闭http客户端"""
        self._keepalive_stop_event.set()
        if self._keepalive_task:
            await asyncio.gather(self._keepalive_task, return_exceptions=True)

        await self.http_client.aclose()
        logger.info("LLM TTS 插件已卸载，HTTP客户端已关闭。")
