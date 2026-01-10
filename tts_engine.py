import asyncio
import os
import re
import uuid
import wave
import time
from pathlib import Path
from typing import Optional, List, Dict

import httpx
from astrbot.api import logger, AstrBotConfig

# --- 音频参数 (必须与Genie TTS服务输出匹配) ---
BYTES_PER_SAMPLE = 2
CHANNELS = 1
SAMPLE_RATE = 32000


class TTSEngine:
    """处理所有与TTS合成相关的核心逻辑，包括文本分块、并发合成和音频合并"""

    def __init__(
        self,
        config: AstrBotConfig,
        http_client: httpx.AsyncClient,
        plugin_data_dir: Path,
    ):
        self.config = config
        self.http_client = http_client
        self.plugin_data_dir = plugin_data_dir
        self.tts_server_index = 0

        # 服务器锁：确保同一服务器的 set_reference_audio + /tts 是原子操作
        self._server_locks: Dict[str, asyncio.Lock] = {}

        # 全局合成锁：确保多个语音请求按"先到先得"顺序串行处理
        self._synthesis_lock = asyncio.Lock()

        # 设置临时音频目录
        self.temp_audio_dir = self.plugin_data_dir / "temp_audio"
        self.temp_audio_dir.mkdir(parents=True, exist_ok=True)

        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """定期清理过期的临时音频文件"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时检查一次
                current_time = time.time()
                expiration_time = 1800  # 30分钟过期

                if not self.temp_audio_dir.exists():
                    continue

                count = 0
                for file_path in self.temp_audio_dir.glob("*.wav"):
                    try:
                        if current_time - file_path.stat().st_mtime > expiration_time:
                            file_path.unlink()
                            count += 1
                    except Exception as e:
                        logger.warning(f"清理文件 {file_path} 失败: {e}")

                if count > 0:
                    logger.info(f"已清理 {count} 个过期的临时音频文件。")

            except Exception as e:
                logger.error(f"清理任务发生错误: {e}")

    def _split_text_into_chunks(self, text: str, sentences_per_chunk: int) -> list[str]:
        """根据标点将文本切分为句子，再按指定数量合并成块"""
        if sentences_per_chunk <= 0:
            return [text]

        regex_pattern = self.config.get("sentence_split_regex", r"([。、，！？,.!?])")

        sentences = re.split(regex_pattern, text)
        if not sentences:
            return []

        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            delimiter = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence:
                full_sentences.append(sentence + delimiter)
        if len(sentences) % 2 == 1 and sentences[-1]:
            full_sentences.append(sentences[-1])

        chunks = []
        for i in range(0, len(full_sentences), sentences_per_chunk):
            chunk = "".join(full_sentences[i : i + sentences_per_chunk])
            chunks.append(chunk)

        logger.info(f"文本已切分为 {len(chunks)} 个块。")
        return chunks

    async def _merge_wav_files(self, input_paths: list[str]) -> Optional[str]:
        """以无损的方式将多个WAV文件按顺序合并为一个，并清理分块文件。"""
        if not input_paths:
            return None

        output_path = self.temp_audio_dir / f"{uuid.uuid4()}_merged.wav"

        try:
            with wave.open(input_paths[0], "rb") as wf_in:
                params = wf_in.getparams()

            with wave.open(str(output_path), "wb") as wf_out:
                wf_out.setparams(params)
                for file_path in input_paths:
                    with wave.open(file_path, "rb") as wf_in:
                        wf_out.writeframes(wf_in.readframes(wf_in.getnframes()))

            logger.info(f"成功将 {len(input_paths)} 个音频文件合并到: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"合并WAV文件时出错: {e}")
            return None
        finally:
            # 无论合并成功与否，都尝试清理输入的临时分块文件
            for file_path in input_paths:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except OSError as e:
                    logger.warning(f"删除临时文件 {file_path} 失败: {e}")

    def _get_server_lock(self, server_url: str) -> asyncio.Lock:
        """获取或创建指定服务器的锁，用于保证 set_reference_audio + /tts 的原子性。"""
        if server_url not in self._server_locks:
            self._server_locks[server_url] = asyncio.Lock()
        return self._server_locks[server_url]

    async def _attempt_synthesis_on_server(
        self,
        server_url: str,
        character_name: str,
        ref_audio_path: str,
        ref_audio_text: str,
        text: str,
        session_id_for_log: str,
        language: str = None,
    ) -> Optional[str]:
        """使用单个指定的TTS服务器尝试合成语音，并返回保存好的文件路径。"""
        logger.info(f"[{session_id_for_log}] 尝试TTS服务器: {server_url}")

        # 获取此服务器的锁，确保 set_reference_audio 和 /tts 不会被其他请求打断
        server_lock = self._get_server_lock(server_url)

        async with server_lock:
            try:
                # Propagate the language parameter directly, fallback to config only if None provided
                if not language:
                    language = self.config.get("tts_default_language", "jp")

                ref_payload = {
                    "character_name": character_name,
                    "audio_path": ref_audio_path,
                    "audio_text": ref_audio_text,
                    "language": language,
                }
                tts_timeout = self.config.get("tts_timeout_seconds", 120)
                response = await self.http_client.post(
                    f"{server_url}/set_reference_audio", json=ref_payload, timeout=60
                )
                response.raise_for_status()

                tts_payload = {
                    "character_name": character_name,
                    "text": text,
                    "split_sentence": True,
                }
                async with self.http_client.stream(
                    "POST", f"{server_url}/tts", json=tts_payload, timeout=tts_timeout
                ) as response_tts:
                    response_tts.raise_for_status()
                    output_path = self.temp_audio_dir / f"{uuid.uuid4()}.wav"
                    with wave.open(str(output_path), "wb") as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(BYTES_PER_SAMPLE)
                        wf.setframerate(SAMPLE_RATE)
                        async for chunk in response_tts.aiter_bytes():
                            wf.writeframes(chunk)
                    return str(output_path)
            except Exception as e:
                logger.warning(
                    f"[{session_id_for_log}] TTS服务器 {server_url} 交互失败: {e}"
                )
                return None

    async def _synthesis_worker(
        self,
        worker_id: int,
        task_queue: asyncio.Queue,
        results_list: list,
        retry_counts: dict,
        character_name: str,
        ref_audio_path: str,
        ref_audio_text: str,
        session_id_for_log: str,
        language: str = None,
    ):
        """单个TTS服务器的工作进程，从队列中获取任务并处理，失败时会重试"""
        servers = self.config.get("tts_servers", [])
        num_servers = len(servers)
        max_retries = self.config.get("tts_max_retries", 3)

        while True:
            try:
                # 使用非阻塞方式获取任务，如果队列为空则退出
                task_index, chunk_text = task_queue.get_nowait()
            except asyncio.QueueEmpty:
                # 队列真正为空，退出循环
                break
            except asyncio.CancelledError:
                break

            current_retry = retry_counts.get(task_index, 0)

            # 智能服务器选择：优先尝试获取空闲服务器的锁
            start_server_idx = worker_id % num_servers
            audio_path = None

            # 第一轮：尝试非阻塞获取锁，找到空闲服务器立即使用
            for i in range(num_servers):
                server_idx = (start_server_idx + i) % num_servers
                server_url = servers[server_idx].strip("/")
                server_lock = self._get_server_lock(server_url)

                if not server_lock.locked():
                    # 服务器空闲，尝试合成
                    log_id = f"{session_id_for_log}-chunk-{task_index + 1}"
                    audio_path = await self._attempt_synthesis_on_server(
                        server_url,
                        character_name,
                        ref_audio_path,
                        ref_audio_text,
                        chunk_text,
                        log_id,
                        language=language,
                    )
                    if audio_path:
                        logger.info(
                            f"[Worker-{worker_id}] 成功合成块 {task_index + 1} 于服务器 {server_url}"
                        )
                        results_list[task_index] = audio_path
                        break

            # 第二轮：如果所有服务器都忙，则等待第一个可用的服务器
            if not audio_path:
                for i in range(num_servers):
                    server_idx = (start_server_idx + i) % num_servers
                    server_url = servers[server_idx].strip("/")
                    log_id = f"{session_id_for_log}-chunk-{task_index + 1}"

                    audio_path = await self._attempt_synthesis_on_server(
                        server_url,
                        character_name,
                        ref_audio_path,
                        ref_audio_text,
                        chunk_text,
                        log_id,
                        language=language,
                    )
                    if audio_path:
                        logger.info(
                            f"[Worker-{worker_id}] 成功合成块 {task_index + 1} 于服务器 {server_url} (等待后)"
                        )
                        results_list[task_index] = audio_path
                        break

            if not audio_path:
                # 失败处理：检查是否还有重试机会
                if current_retry < max_retries:
                    retry_counts[task_index] = current_retry + 1
                    logger.warning(
                        f"[Worker-{worker_id}] 块 {task_index + 1} 合成失败，放回队列重试 ({current_retry + 1}/{max_retries})"
                    )
                    # 延迟一小段时间后重新放回队列，避免立即重试造成服务器过载
                    await asyncio.sleep(0.5)
                    await task_queue.put((task_index, chunk_text))
                else:
                    logger.error(
                        f"[Worker-{worker_id}] 块 {task_index + 1} 达到最大重试次数 ({max_retries})，放弃。"
                    )
                    results_list[task_index] = None

            task_queue.task_done()

    async def synthesize(
        self,
        character_name: str,
        ref_audio_path: str,
        ref_audio_text: str,
        text: str,
        session_id_for_log: str,
        language: str = None,
    ) -> Optional[str]:
        """执行语音合成的核心入口点，支持并发处理。使用全局锁确保先到先得。"""

        # 全局锁确保语音请求按顺序处理，先请求的先完成
        async with self._synthesis_lock:
            servers = self.config.get("tts_servers", [])
            if not servers:
                logger.error(f"[{session_id_for_log}] 未配置TTS服务器。")
                return None

            if self.config.get("enable_sentence_splitting", False):
                sentences_per_chunk = self.config.get("sentences_per_chunk", 2)
                text_chunks = self._split_text_into_chunks(text, sentences_per_chunk)

                if len(text_chunks) > 1:
                    task_queue = asyncio.Queue()
                    for i, chunk in enumerate(text_chunks):
                        task_queue.put_nowait((i, chunk))

                    results_list = [None] * len(text_chunks)
                    retry_counts = {}  # 跟踪每个块的重试次数
                    # Worker数量等于块数，让每个块有专属worker，服务器锁会自然调度执行顺序
                    num_workers = len(text_chunks)
                    workers = [
                        asyncio.create_task(
                            self._synthesis_worker(
                                worker_id=i,
                                task_queue=task_queue,
                                results_list=results_list,
                                retry_counts=retry_counts,
                                character_name=character_name,
                                ref_audio_path=ref_audio_path,
                                ref_audio_text=ref_audio_text,
                                session_id_for_log=session_id_for_log,
                                language=language,
                            )
                        )
                        for i in range(num_workers)
                    ]

                    logger.info(
                        f"[{session_id_for_log}] 创建了 {num_workers} 个worker来处理 {len(text_chunks)} 个语音块..."
                    )
                    await task_queue.join()
                    for worker in workers:
                        worker.cancel()
                    await asyncio.gather(*workers, return_exceptions=True)

                    successful_paths = [path for path in results_list if path]

                    # 如果有部分失败，或者全部失败，都需要清理已经生成的临时文件
                    if len(successful_paths) != len(text_chunks):
                        logger.error(
                            f"[{session_id_for_log}] 部分或全部语音块合成失败，正在清理临时文件。"
                        )
                        for path in successful_paths:
                            try:
                                if os.path.exists(path):
                                    os.remove(path)
                            except Exception as e:
                                logger.warning(f"清理残留文件 {path} 失败: {e}")
                        return None

                    return (
                        successful_paths[0]
                        if len(successful_paths) == 1
                        else await self._merge_wav_files(successful_paths)
                    )

            # 如果不切分，则使用轮询逻辑
            logger.info(f"[{session_id_for_log}] 使用单块模式进行合成。")
            start_index = self.tts_server_index
            for i in range(len(servers)):
                current_index = (start_index + i) % len(servers)
                server_url = servers[current_index].strip("/")

                if i == 0:
                    self.tts_server_index = (start_index + 1) % len(servers)

                audio_path = await self._attempt_synthesis_on_server(
                    server_url=server_url,
                    character_name=character_name,
                    ref_audio_path=ref_audio_path,
                    ref_audio_text=ref_audio_text,
                    text=text,
                    session_id_for_log=session_id_for_log,
                    language=language,
                )
                if audio_path:
                    return audio_path

            logger.error(f"[{session_id_for_log}] 尝试所有TTS服务器后合成失败。")
            return None
