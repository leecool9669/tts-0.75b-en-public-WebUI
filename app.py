from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

import gradio as gr

ROOT = Path(__file__).resolve().parent


def _load_hero_image() -> str | None:
    """返回模型卡片或标签区域的截图路径（如不存在则返回 None）。"""
    hero = ROOT / "images" / "tts_075b_hf_page.png"
    if hero.exists():
        return str(hero)
    return None


def fake_streaming_tts(
    text: str,
    speaker_hint: str,
    speaking_rate: float,
    prosody_strength: float,
    temperature: float,
    chunk_ms: int,
) -> Tuple[str, str]:
    """占位推理函数：模拟流式 TTS 的工作流程与配置摘要，不生成真实音频。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not text.strip():
        explanation = (
            "（演示模式）请输入一小段英文文本，例如一两句日常对话，用于模拟流式文本到语音合成过程。"
        )
    else:
        explanation = (
            "【演示输出】当前前端仅以文本形式说明合成流程，不调用任何大体积声学模型权重。\n\n"
            f"- 输入文本片段：\n{text.strip()}\n\n"
            "- 在真实系统中，前端会将文本分词、编码为离散 token，"
            "再交由分层 Transformer 骨干网络与深度解码模块逐块生成音频 token，"
            "随后通过神经声码器还原为可播放的波形。\n"
            "- 由于采用流式建模，模型在接收到前几个词后即可输出首批音频帧，"
            "后续词语会以小跨度延迟持续补充到语音序列中。"
        )

    config = (
        f"时间戳：{ts}\n"
        f"说话人提示（speaker hint）：{speaker_hint or '未指定，将采用通用中性音色'}\n"
        f"语速控制（speaking rate）：{speaking_rate:.2f}\n"
        f"韵律强度（prosody strength）：{prosody_strength:.2f}\n"
        f"采样温度（temperature）：{temperature:.2f}\n"
        f"流式分块时长（chunk size）：{chunk_ms} ms\n"
        "说明：本演示版本仅记录参数与流程，不会访问远程仓库或下载实际 TTS 模型权重。"
    )

    return explanation, config


def build_app() -> gr.Blocks:
    hero_path = _load_hero_image()

    with gr.Blocks(
        title="Kyutai TTS-0.75b English Streaming WebUI（演示版）",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # Kyutai TTS-0.75b English Streaming WebUI（演示版）

            本界面用于模拟一个基于分层 Transformer 与延迟流建模思想的英文流式文本到语音合成系统的前端，
            重点展示从文本输入、说话人提示、流式分块到合成结果说明的一整套交互流程。
            当前版本仅在本地浏览器中可视化参数与说明性文本，不加载或运行任何真实的声学模型。
            """
        )

        if hero_path is not None:
            with gr.Row():
                gr.Image(
                    value=hero_path,
                    label="模型卡片与标签区域截图",
                    type="filepath",
                )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 文本与说话人配置")

                text = gr.Textbox(
                    label="待合成英文文本",
                    lines=6,
                    placeholder=(
                        "例如：Hey there, thanks for trying this streaming text-to-speech demo. "
                        "The model starts speaking after just a few words."
                    ),
                )

                speaker_hint = gr.Textbox(
                    label="说话人提示（可选）",
                    placeholder="例如：中性成年女性、较为温和的广播主持、充满活力的技术解说等。",
                )

                with gr.Accordion("高级控制参数", open=False):
                    speaking_rate = gr.Slider(
                        minimum=0.5,
                        maximum=1.5,
                        value=1.0,
                        step=0.05,
                        label="语速控制（0.5=偏慢，1.0=标准，1.5=偏快）",
                    )
                    prosody_strength = gr.Slider(
                        minimum=0.5,
                        maximum=1.5,
                        value=1.0,
                        step=0.05,
                        label="韵律起伏强度（0.5=更平，1.5=更夸张）",
                    )
                    temperature = gr.Slider(
                        minimum=0.3,
                        maximum=1.5,
                        value=0.9,
                        step=0.05,
                        label="采样温度（temperature）",
                    )
                    chunk_ms = gr.Slider(
                        minimum=80,
                        maximum=320,
                        value=160,
                        step=10,
                        label="单次流式推理分块长度（毫秒）",
                    )

                run_btn = gr.Button("开始流式合成（演示模式）", variant="primary")

            with gr.Column(scale=3):
                gr.Markdown("### 合成过程说明与参数摘要")

                explanation = gr.Textbox(
                    label="合成过程与听感预期（说明性文本）",
                    lines=16,
                )

                config = gr.Textbox(
                    label="当前流式推理配置摘要",
                    lines=10,
                )

        run_btn.click(
            fn=fake_streaming_tts,
            inputs=[
                text,
                speaker_hint,
                speaking_rate,
                prosody_strength,
                temperature,
                chunk_ms,
            ],
            outputs=[explanation, config],
        )

        gr.Markdown(
            """
            ---
            本 WebUI 仅作为流式文本到语音模型的可视化前端雏形。
            在后续工程实践中，可以在保持交互结构不变的前提下，将 `fake_streaming_tts` 替换为实际的推理函数，
            将文本编码为离散 token 并调用层次化 Transformer 与神经声码器，实现真正的在线语音合成服务。
            """
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    # 仅在本地启动演示用前端，不暴露公网地址
    app.launch(server_name="127.0.0.1", server_port=7861, share=False)
