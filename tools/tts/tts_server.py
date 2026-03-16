#!/usr/bin/env python3
"""
Streaming TTS Demo Server

Demonstrates streaming vs non-streaming TTS generation time comparison.
Note: The binary outputs complete WAV files, so "first byte" = generation complete.
True streaming would require binary to output to stdout.
"""

import http.server
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path
from urllib.parse import parse_qs

# Configuration
PORT = 8123
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TTS_BINARY = BASE_DIR / "build" / "bin" / "llama-orpheus-tts"
MODEL_PATH = BASE_DIR / "models" / "orpheus-tts" / "orpheus-3b-f16.gguf"
VOCODER_PATH = BASE_DIR / "models" / "snac" / "snac-24khz-f16.gguf"
OUTPUT_DIR = BASE_DIR / "test_output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HTML_PAGE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎵 TTS Demo - Generation Time Comparison</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .buttons {
            display: flex;
            gap: 15px;
            margin: 20px 0;
        }
        button {
            flex: 1;
            padding: 15px 25px;
            font-size: 16px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }
        .btn-both {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }
        .btn-streaming {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-nonstreaming {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(0,0,0,0.2); }
        button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .status {
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            display: none;
        }
        .status.active { display: block; }
        .status.loading { background: #fff3cd; color: #856404; }
        .status.success { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }

        .results {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .result-card {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
        }
        .result-card h3 {
            margin: 0 0 15px 0;
            color: #333;
        }
        .metrics {
            font-size: 14px;
            color: #666;
            margin-bottom: 15px;
        }
        .metrics div {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        .metrics div:last-child { border-bottom: none; }
        .metric-value { font-weight: bold; color: #333; }
        audio {
            width: 100%;
            margin-top: 10px;
        }
        .progress-bar {
            height: 6px;
            background: #e0e0e0;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-bar .progress {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s;
        }

        .comparison {
            background: #e8f5e9;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
            display: none;
        }
        .comparison.active { display: block; }
        .comparison h4 { margin: 0 0 10px 0; color: #2E7D32; }
        .comparison .diff { font-size: 18px; font-weight: bold; color: #1B5E20; }

        .note {
            background: #fff3e0;
            border-left: 4px solid #FF9800;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
            font-size: 14px;
        }
        .note h4 { margin: 0 0 10px 0; color: #E65100; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 TTS 生成时间对比</h1>
        <p class="subtitle">对比流式 vs 非流式语音合成的性能差异</p>

        <div class="note">
            <h4>📝 说明</h4>
            <p>此演示对比两种模式的<strong>总生成时间</strong>：</p>
            <ul>
                <li><strong>流式模式 (--streaming)</strong>: SNAC vocoder 使用流式解码，逐帧生成音频</li>
                <li><strong>非流式模式</strong>: 等待所有 token 后一次性解码全部音频</li>
            </ul>
            <p>注意：当前二进制输出完整 WAV 文件。真正的流式播放需要修改二进制支持 stdout 输出。</p>
        </div>

        <textarea id="textInput" placeholder="输入要合成的文本..."></textarea>

        <div class="buttons">
            <button class="btn-both" onclick="generateBoth()">
                ⚡ 同时生成对比
            </button>
            <button class="btn-streaming" onclick="generateStreaming()">
                🚀 流式生成
            </button>
            <button class="btn-nonstreaming" onclick="generateNonStreaming()">
                📦 非流式生成
            </button>
        </div>

        <div id="status" class="status"></div>

        <div id="comparison" class="comparison">
            <h4>📊 性能对比</h4>
            <div id="comparisonContent"></div>
        </div>

        <div class="results">
            <div class="result-card">
                <h3>🚀 流式模式</h3>
                <div class="metrics" id="streamingMetrics">
                    <div><span>生成时间:</span> <span class="metric-value" id="streamTime">--</span></div>
                    <div><span>音频时长:</span> <span class="metric-value" id="streamDuration">--</span></div>
                    <div><span>实时率:</span> <span class="metric-value" id="streamRTF">--</span></div>
                </div>
                <div class="progress-bar"><div class="progress" id="streamProgress"></div></div>
                <audio id="streamingAudio" controls></audio>
            </div>

            <div class="result-card">
                <h3>📦 非流式模式</h3>
                <div class="metrics" id="nonstreamingMetrics">
                    <div><span>生成时间:</span> <span class="metric-value" id="nonStreamTime">--</span></div>
                    <div><span>音频时长:</span> <span class="metric-value" id="nonStreamDuration">--</span></div>
                    <div><span>实时率:</span> <span class="metric-value" id="nonStreamRTF">--</span></div>
                </div>
                <div class="progress-bar"><div class="progress" id="nonStreamProgress"></div></div>
                <audio id="nonStreamingAudio" controls></audio>
            </div>
        </div>
    </div>

    <script>
        const textInput = document.getElementById('textInput');
        const statusDiv = document.getElementById('status');
        let streamResult = null;
        let nonStreamResult = null;

        function showStatus(message, type) {
            statusDiv.textContent = message;
            statusDiv.className = 'status active ' + type;
        }

        function hideStatus() {
            statusDiv.className = 'status';
        }

        function formatTime(seconds) {
            if (seconds === null || seconds === undefined) return '--';
            return seconds.toFixed(2) + 's';
        }

        function formatRTF(rtf) {
            if (rtf === null || rtf === undefined) return '--';
            return rtf.toFixed(3) + 'x';
        }

        async function generateTTS(streaming) {
            const text = textInput.value.trim();
            if (!text) {
                showStatus('请输入文本！', 'error');
                return null;
            }

            const endpoint = streaming ? '/tts/streaming' : '/tts/nonstreaming';
            const audioElement = document.getElementById(streaming ? 'streamingAudio' : 'nonStreamingAudio');
            const progressElement = document.getElementById(streaming ? 'streamProgress' : 'nonStreamProgress');
            const timeElement = document.getElementById(streaming ? 'streamTime' : 'nonStreamTime');
            const durationElement = document.getElementById(streaming ? 'streamDuration' : 'nonStreamDuration');
            const rtfElement = document.getElementById(streaming ? 'streamRTF' : 'nonStreamRTF');

            // Reset
            timeElement.textContent = '--';
            durationElement.textContent = '--';
            rtfElement.textContent = '--';
            progressElement.style.width = '0%';
            audioElement.src = '';

            showStatus((streaming ? '流式' : '非流式') + '生成中...', 'loading');

            const startTime = Date.now();

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text, voice: 'tara' })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const blob = await response.blob();
                const endTime = Date.now();
                const genTime = (endTime - startTime) / 1000;

                // Get metadata from headers
                const audioDuration = parseFloat(response.headers.get('X-Audio-Duration') || '0');
                const rtf = audioDuration > 0 ? genTime / audioDuration : 0;

                // Update UI
                timeElement.textContent = formatTime(genTime);
                durationElement.textContent = formatTime(audioDuration);
                rtfElement.textContent = formatRTF(rtf);
                progressElement.style.width = '100%';

                // Play audio
                audioElement.src = URL.createObjectURL(blob);

                showStatus(`${streaming ? '流式' : '非流式'}生成完成！`, 'success');

                return { genTime, audioDuration, rtf };

            } catch (error) {
                showStatus('错误: ' + error.message, 'error');
                console.error(error);
                return null;
            }
        }

        function updateComparison() {
            if (streamResult && nonStreamResult) {
                const diff = nonStreamResult.genTime - streamResult.genTime;
                const faster = diff > 0 ? '流式' : '非流式';
                const absDiff = Math.abs(diff);

                const comparisonDiv = document.getElementById('comparison');
                const contentDiv = document.getElementById('comparisonContent');

                if (diff > 0) {
                    contentDiv.innerHTML = `
                        <p><strong>${faster}</strong> 模式更快 <span class="diff">${absDiff.toFixed(2)}s</span></p>
                        <p>流式: ${streamResult.genTime.toFixed(2)}s | 非流式: ${nonStreamResult.genTime.toFixed(2)}s</p>
                    `;
                } else {
                    contentDiv.innerHTML = `
                        <p><strong>${faster}</strong> 模式更快 <span class="diff">${absDiff.toFixed(2)}s</span></p>
                        <p>流式: ${streamResult.genTime.toFixed(2)}s | 非流式: ${nonStreamResult.genTime.toFixed(2)}s</p>
                    `;
                }
                comparisonDiv.classList.add('active');
            }
        }

        async function generateStreaming() {
            streamResult = await generateTTS(true);
            updateComparison();
            setTimeout(hideStatus, 2000);
        }

        async function generateNonStreaming() {
            nonStreamResult = await generateTTS(false);
            updateComparison();
            setTimeout(hideStatus, 2000);
        }

        async function generateBoth() {
            streamResult = null;
            nonStreamResult = null;
            document.getElementById('comparison').classList.remove('active');

            showStatus('同时启动两种模式生成...', 'loading');

            // Run both in parallel
            const [sResult, nsResult] = await Promise.all([
                generateTTS(true),
                generateTTS(false)
            ]);

            streamResult = sResult;
            nonStreamResult = nsResult;
            updateComparison();
            setTimeout(hideStatus, 2000);
        }

        // Set default text
        textInput.value = "The quick brown fox jumps over the lazy dog. This is a test of the text to speech system. Machine learning has made significant progress in recent years.";
    </script>
</body>
</html>
'''


class TTSServer(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")

    def send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        else:
            self.send_error(404)

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            text = data.get('text', '')
            voice = data.get('voice', 'tara')

            if not text:
                self.send_error(400, 'Missing text parameter')
                return

            streaming = self.path == '/tts/streaming'
            mode_str = 'streaming' if streaming else 'non-streaming'
            print(f"\n{'='*60}")
            print(f"TTS Request: {mode_str} mode")
            print(f"Text ({len(text)} chars): {text[:80]}...")

            output_file = OUTPUT_DIR / f"demo_{uuid.uuid4().hex[:8]}.wav"

            cmd = [
                str(TTS_BINARY),
                '-m', str(MODEL_PATH),
                '--model-vocoder', str(VOCODER_PATH),
                '-p', text,
                '-o', str(output_file),
                '--use-snac-ggml'
            ]

            if streaming:
                cmd.append('--streaming')

            print(f"Running: {' '.join(cmd[:5])}...")

            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            end_time = time.time()
            gen_time = end_time - start_time

            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                self.send_error(500, f'TTS generation failed: {result.stderr[:500]}')
                return

            audio_duration = self.get_wav_duration(output_file)
            rtf = gen_time / audio_duration if audio_duration > 0 else 0

            print(f"Generated: {output_file.name}")
            print(f"Generation time: {gen_time:.2f}s")
            print(f"Audio duration: {audio_duration:.2f}s")
            print(f"RTF (Real-Time Factor): {rtf:.3f}x")

            self.send_response(200)
            self.send_header('Content-Type', 'audio/wav')
            self.send_header('X-Audio-Duration', str(audio_duration))
            self.send_cors_headers()
            self.end_headers()

            with open(output_file, 'rb') as f:
                self.wfile.write(f.read())

            try:
                output_file.unlink()
            except:
                pass

        except subprocess.TimeoutExpired:
            self.send_error(504, 'TTS generation timeout')
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON')
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.send_error(500, str(e))

    def get_wav_duration(self, wav_path):
        try:
            import wave
            with wave.open(str(wav_path), 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        except:
            return 0


def main():
    print("="*60)
    print("TTS Demo Server")
    print("="*60)
    print(f"Server: http://localhost:{PORT}")
    print(f"TTS Binary: {TTS_BINARY}")
    print(f"Model: {MODEL_PATH}")
    print(f"Vocoder: {VOCODER_PATH}")
    print("="*60)

    if not TTS_BINARY.exists():
        print(f"ERROR: TTS binary not found at {TTS_BINARY}")
        return

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    if not VOCODER_PATH.exists():
        print(f"ERROR: Vocoder not found at {VOCODER_PATH}")
        return

    print("\nStarting server... Press Ctrl+C to stop.\n")

    server = http.server.HTTPServer(('', PORT), TTSServer)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == '__main__':
    main()
