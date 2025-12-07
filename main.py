import asyncio
import httpx
import os
import re
from typing import Dict, Optional, Set

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger, AstrBotConfig
import astrbot.api.message_components as Comp
from astrbot.api.provider import LLMResponse, ProviderRequest

# 从新模块导入功能
from .emotion_manager import EmotionManager
from .tts_engine import TTSEngine
from .external_apis import translate_text


@register(
    "astrbot_plugin_tts_llm",
    "clown145",
    "一个通过LLM、翻译和TTS实现语音合成的插件",
    "1.3.1",
    "https://github.com/clown145/astrbot_plugin_tts_llm",
)
class LlmTtsPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.active_sessions: Set[str] = set()
        self.w_active_sessions: Set[str] = set()
        self.active_groups: Set[str] = set() # 新增：群组级TTS开关
        self.session_emotions: Dict[str, Dict[str, str]] = {}
        self.session_w_settings: Dict[str, Dict[str, str]] = {}
        self._keepalive_stop_event = asyncio.Event()
        self._keepalive_task: Optional[asyncio.Task] = None

        # 初始化辅助模块
        plugin_data_dir = StarTools.get_data_dir("astrbot_plugin_tts_llm")
        emotions_file_path = plugin_data_dir / "emotions.json"
        self.emotion_manager = EmotionManager(emotions_file_path)
        
        self.http_client = httpx.AsyncClient(timeout=300.0)
        self.tts_engine = TTSEngine(self.config, self.http_client, plugin_data_dir)

        if self.config.get("enable_space_keepalive"):
            self._keepalive_task = asyncio.create_task(self._keep_alive_loop())

        logger.info("LLM TTS 插件已加载。")

    def _get_keepalive_urls(self) -> list[str]:
        """获取所有需要保活的目标地址。包括配置的TTS服务器和额外的保活地址。"""
        urls = set()
        
        # 添加所有配置的TTS服务器
        servers = self.config.get("tts_servers", [])
        if servers:
            for server in servers:
                if isinstance(server, str) and server:
                    urls.add(server.rstrip("/"))

        # 添加额外配置的保活地址
        custom_url = self.config.get("space_keepalive_url")
        if custom_url:
            urls.add(custom_url.rstrip("/"))

        return list(urls)

    async def _keep_alive_loop(self):
        """定时ping所有目标地址以避免休眠。"""
        interval_minutes = max(int(self.config.get("space_keepalive_interval_minutes", 25)), 1)

        async def ping(url):
            try:
                response = await self.http_client.get(url, timeout=30)
                logger.info(f"保活请求已发送到 {url}，状态码: {response.status_code}")
            except Exception as exc:
                logger.warning(f"向 {url} 发送保活请求失败: {exc}")

        while not self._keepalive_stop_event.is_set():
            try:
                target_urls = self._get_keepalive_urls()
                if not target_urls:
                    logger.warning("未找到任何可用于保活的地址，已跳过本次保活任务。")
                else:
                    await asyncio.gather(*(ping(url) for url in target_urls))
            except Exception as e:
                logger.error(f"保活任务发生意外错误: {e}")

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

    @filter.command("ttg", alias={"开启群语音"})
    async def start_group_tts(self, event: AstrMessageEvent):
        """开启当前群组的语音合成 (对所有人生效)"""
        group_id = event.message_obj.group_id
        if not group_id:
             yield event.plain_result("❌ 此指令仅限群聊使用。")
             return
        
        self.active_groups.add(group_id)
        default_char = self.config.get("default_character")
        default_emotion = self.config.get("default_emotion_name")
        
        settings = self.config.get("llm_injection_settings", {})
        enable_emotion = settings.get("enable_llm_emotion", False)
        
        logger.info(f"群组 [{group_id}] 的 LLM TTS 功能已开启。")
        
        if enable_emotion:
            yield event.plain_result(f"▶️ 本群组的LLM语音合成已开启 (全员生效)。\n当前已启用LLM情感注入，情感将由AI自动决定。\n(默认保底情感: {default_char} - {default_emotion})")
        else:
            yield event.plain_result(f"▶️ 本群组的LLM语音合成已开启 (全员生效)。\n当前为固定情感模式: {default_char} - {default_emotion}")

    @filter.command("ttg-q", alias={"关闭群语音"})
    async def stop_group_tts(self, event: AstrMessageEvent):
        """关闭当前群组的语音合成"""
        group_id = event.message_obj.group_id
        if not group_id:
             yield event.plain_result("❌ 此指令仅限群聊使用。")
             return

        self.active_groups.discard(group_id)
        logger.info(f"群组 [{group_id}] 的 LLM TTS 功能已关闭。")
        yield event.plain_result("⏹️ 本群组的LLM语音合成已关闭。")

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

    @filter.on_llm_request()
    async def inject_llm_prompt(self, event: AstrMessageEvent, req: ProviderRequest):
        """在LLM请求前注入提示词"""
        session_id = event.unified_msg_origin
        group_id = event.message_obj.group_id
        
        # 只有在开启了TTS模式（自动或固定，或群组模式）时才注入
        is_active = (session_id in self.active_sessions or 
                     session_id in self.w_active_sessions or 
                     (group_id and group_id in self.active_groups))
        
        if not is_active:
            return

        settings = self.config.get("llm_injection_settings", {})
        enable_emotion = settings.get("enable_llm_emotion", False)
        enable_translation = settings.get("enable_llm_translation", False)

        if not enable_emotion and not enable_translation:
            return

        prompts_to_inject = []

        if enable_emotion:
            # 确定当前角色以获取可用情感列表
            char_name = None
            if session_id in self.w_active_sessions:
                char_name = self.session_w_settings.get(session_id, {}).get("character") or self.config.get("default_character")
            elif session_id in self.active_sessions or (group_id and group_id in self.active_groups):
                # 固定模式或群组模式下
                session_setting = self.session_emotions.get(session_id)
                char_name = session_setting["character"] if session_setting else self.config.get("default_character")

            if char_name and self.emotion_manager.character_exists(char_name):
                emotions = list(self.emotion_manager.emotions_data[char_name].keys())
                emotions_str = ", ".join(emotions)
                
                prompt_template = settings.get("llm_emotion_prompt", "")
                try:
                    emotion_prompt = prompt_template.format(emotions=emotions_str)
                except KeyError:
                    emotion_prompt = prompt_template
                prompts_to_inject.append(emotion_prompt)

        if enable_translation:
            trans_prompt = settings.get("llm_translation_prompt", "")
            if trans_prompt:
                prompts_to_inject.append(trans_prompt)

        if prompts_to_inject:
            final_prompt = "\n\n".join(prompts_to_inject)
            req.system_prompt += f"\n\n{final_prompt}"
            logger.info(f"[{session_id}] 已注入LLM提示词 (Emotion: {enable_emotion}, Trans: {enable_translation})")


    @filter.on_llm_response()
    async def intercept_llm_response_for_tts(self, event: AstrMessageEvent, resp: LLMResponse):
        session_id = event.unified_msg_origin
        group_id = event.message_obj.group_id
        original_text = resp.completion_text.strip()
        if not original_text:
            return

        # 0. 清理可能存在的幻觉报错 (防止LLM复读之前的错误提示)
        original_text = original_text.replace("(TTS失败: 翻译无结果)", "")
        original_text = original_text.replace("(TTS合成失败)", "")
        original_text = original_text.replace("(TTS失败: 角色", "") # 模糊匹配

        # 检查是否开启了TTS (个人会话 或 群组)
        is_active = (session_id in self.active_sessions or 
                     session_id in self.w_active_sessions or 
                     (group_id and group_id in self.active_groups))

        if not is_active:
            return

        settings = self.config.get("llm_injection_settings", {})
        enable_llm_emotion = settings.get("enable_llm_emotion", False)
        enable_llm_translation = settings.get("enable_llm_translation", False)

        # 1. 提取情感标签 [emotion=xxx]
        emotion_match = re.search(r'\[emotion=(.*?)\]', original_text)
        injected_emotion = None
        if emotion_match:
            injected_emotion = emotion_match.group(1).strip()
            # 从原文中移除标签，保持回复干净
            original_text = original_text.replace(emotion_match.group(0), "")

        # 2. 提取翻译内容 $xxx$ 或 ＄xxx＄
        # 兼容全角符号，且支持多行匹配
        translation_match = re.search(r'[\\$＄](.*?)[\\$＄]', original_text, re.DOTALL)
        injected_translation = None
        if translation_match:
            injected_translation = translation_match.group(1).strip()
            # 从原文中移除翻译，保持回复干净
            original_text = original_text.replace(translation_match.group(0), "")

        # 更新 LLM 回复文本为净化后的文本 (去除标签和翻译部分)
        resp.completion_text = original_text.strip()
        # 同时更新 result_chain 中的 Plain 消息，否则用户还是会看到标签
        # 注意：这里假设 result_chain 第一个是 Plain。如果不是，可能需要遍历。
        # 简单起见，我们重建 chain
        resp.result_chain.chain = [Comp.Plain(resp.completion_text)]

        # --- 开始 TTS 处理流程 ---
        
        audio_path: Optional[str] = None
        target_emotion = None
        target_text = None
        char_name = None

        # 确定角色
        if session_id in self.w_active_sessions:
            char_name = self.session_w_settings.get(session_id, {}).get("character") or self.config.get("default_character")
        else:
            # 固定模式 或 群组模式
            session_setting = self.session_emotions.get(session_id)
            char_name = session_setting["character"] if session_setting else self.config.get("default_character")
            # 固定模式下，如果没有注入情感，使用默认情感
            if not injected_emotion:
                target_emotion = session_setting["emotion"] if session_setting else self.config.get("default_emotion_name")

        if not char_name or not self.emotion_manager.character_exists(char_name):
            resp.result_chain.chain.append(Comp.Plain(f"\n(TTS失败: 角色'{char_name}'无效)"))
            return

        # 确定情感
        if enable_llm_emotion and injected_emotion:
            target_emotion = injected_emotion
        
        # 确定翻译文本
        if enable_llm_translation and injected_translation:
            target_text = injected_translation
        else:
            # 需要翻译
            # 1. 检查是否使用 AstrBot Provider
            use_astrbot_provider = settings.get("use_astrbot_provider", False)
            provider_id = settings.get("astrbot_provider_id")
            
            if use_astrbot_provider and provider_id:
                try:
                    provider = self.context.get_provider_by_id(provider_id)
                    if provider:
                        trans_prompt = settings.get("translation_prompt", "Translate to Japanese.")
                        llm_resp = await provider.text_chat(
                            prompt=original_text,
                            system_prompt=trans_prompt
                        )
                        target_text = llm_resp.completion_text
                    else:
                        logger.error(f"未找到 Provider ID: {provider_id}")
                except Exception as e:
                    logger.error(f"AstrBot Provider 翻译失败: {e}")

            # 2. 如果还没拿到文本，尝试旧版 API
            if not target_text:
                api_config = self.config.get("translation_api", {})
                # 如果是在 w 模式下且没有注入情感，我们需要同时获取情感和翻译 (旧逻辑)
                if session_id in self.w_active_sessions and not target_emotion:
                     # 旧的自动情感识别逻辑
                    character_emotions = self.emotion_manager.emotions_data[char_name]
                    w_prompt_template = api_config.get("w_mode_prompt")
                    if w_prompt_template:
                        emotion_list_str = ", ".join(character_emotions.keys())
                        augmented_prompt = w_prompt_template.format(emotion_list=emotion_list_str, text=original_text)
                        japanese_text_with_emotion = await translate_text(augmented_prompt, self.http_client, api_config, w_prompt_template)
                        
                        if japanese_text_with_emotion:
                            match = re.search(r'(.*)\[(.+?)\]\s*$', japanese_text_with_emotion.strip(), re.DOTALL)
                            if match:
                                target_text, target_emotion = match.group(1).strip(), match.group(2).strip()
                
                # 普通翻译逻辑
                if not target_text:
                    target_text = await translate_text(original_text, self.http_client, api_config)

        if not target_text:
            resp.result_chain.chain.append(Comp.Plain("\n(TTS失败: 翻译无结果)"))
            return

        # 最终合成
        # 如果此时还没有 target_emotion (比如固定模式没注入，或者自动模式失败)，使用默认
        if not target_emotion:
             target_emotion = self.config.get("default_emotion_name")

        emotion_data = self.emotion_manager.get_emotion_data(char_name, target_emotion)
        if not emotion_data:
             # 尝试回落到默认情感
             default_emotion = self.config.get("default_emotion_name")
             emotion_data = self.emotion_manager.get_emotion_data(char_name, default_emotion)
             if not emotion_data:
                resp.result_chain.chain.append(Comp.Plain(f"\n(TTS失败: 情感'{target_emotion}'无效)"))
                return

        audio_path = await self.tts_engine.synthesize(
            character_name=char_name, ref_audio_path=emotion_data["ref_audio_path"],
            ref_audio_text=emotion_data["ref_audio_text"], text=target_text, session_id_for_log=session_id
        )
        
        if audio_path:
            # 将音频插入到消息链最前面
            resp.result_chain.chain.insert(0, Comp.Record(file=audio_path))
        else:
            resp.result_chain.chain.append(Comp.Plain("\n(TTS合成失败)"))

    async def terminate(self):
        """插件卸载/停用时关闭http客户端"""
        self._keepalive_stop_event.set()
        if self._keepalive_task:
            await asyncio.gather(self._keepalive_task, return_exceptions=True)

        await self.http_client.aclose()
        logger.info("LLM TTS 插件已卸载，HTTP客户端已关闭。")
