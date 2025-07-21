import discord
import asyncio
import aiohttp
import os
import uuid
import json
import random
from io import BytesIO
from discord.ext import commands
from discord import app_commands


class ImageGenCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.session = aiohttp.ClientSession()
        self.server_address = os.getenv("COMFYUI_SERVER_ADDRESS", "127.0.0.1:8188")
        self.client_id = str(uuid.uuid4())

    async def cog_unload(self):
        await self.session.close()

    async def queue_prompt(self, prompt_workflow):
        """ComfyUI에 프롬프트를 전송하고 웹소켓으로 결과를 수신합니다."""
        prompt_url = f"http://{self.server_address}/prompt"
        ws_url = f"ws://{self.server_address}/ws?clientId={self.client_id}"

        # 1. 프롬프트 전송 (HTTP POST)
        async with self.session.post(prompt_url, json={'prompt': prompt_workflow, 'client_id': self.client_id}) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"ComfyUI API 에러: {resp.status} - {error_text}")
            queue_data = await resp.json()
            prompt_id = queue_data['prompt_id']

        # 2. 웹소켓 연결 및 결과 대기
        async with self.session.ws_connect(ws_url) as ws:
            while True:
                msg = await ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    message = json.loads(msg.data)
                    # 'executed' 타입의 메시지이고, 전송한 프롬프트 ID와 일치하는 경우
                    if message['type'] == 'executed' and message['data']['prompt_id'] == prompt_id:
                        outputs = message['data']['output']['images']
                        return outputs
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        return None

    async def get_image(self, filename, subfolder, folder_type):
        """/view 엔드포인트를 통해 생성된 이미지를 가져옵니다."""
        view_url = f"http://{self.server_address}/view"
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        async with self.session.get(view_url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"이미지 다운로드 실패: {resp.status}")
            return await resp.read()

    @app_commands.command(name="generate_image", description="ComfyUI를 사용하여 이미지를 생성합니다.")
    @app_commands.describe(
        positive_prompt="이미지에 포함하고 싶은 내용을 입력하세요 (긍정 프롬프트).",
        negative_prompt="이미지에 포함하고 싶지 않은 내용을 입력하세요 (부정 프롬프트, 선택 사항)."
    )
    async def generate_image(self, interaction: discord.Interaction, positive_prompt: str, negative_prompt: str = ""):
        await interaction.response.defer(ephemeral=False)
        print(f"Positive Prompt: {positive_prompt}")
        positive_prompt = positive_prompt.join("masterpiece, best quality, very awa, newest, recent,")  # 프롬프트를 조합합니다.
        if negative_prompt != "":
            print(f"Negative Prompt: {negative_prompt}")
        negative_prompt = negative_prompt.join("worst quality, worst displeasing, bad anatomy, mosaic censoring, censored, bar censor, watermark, username, signature, twitter username, closed eyes, chibi, deformed,")

        try:
            # SDXL 워크플로우 정의
            # 각 노드는 ComfyUI의 빌딩 블록에 해당합니다.
            prompt_workflow = {
                # 4: 모델 로더. 사용할 SDXL 체크포인트 파일을 지정합니다.
                "4": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "noobaiXLNAIXL_vPred10Version.safetensors"} # 사용하시는 모델 파일명으로 변경하세요.
                },
                # 6: 긍정 프롬프트 인코더 (SDXL용)
                "6": {
                    "class_type": "CLIPTextEncodeSDXL",
                    "inputs": {
                        "width": 1024, "height": 1536, "crop_w": 0, "crop_h": 0,
                        "target_width": 1024, "target_height": 1536,
                        "text_g": positive_prompt, "text_l": positive_prompt,
                        "clip": ["4", 1] # 모델 로더(4)의 CLIP을 사용
                    }
                },
                # 7: 부정 프롬프트 인코더 (SDXL용)
                "7": {
                    "class_type": "CLIPTextEncodeSDXL",
                    "inputs": {
                        "width": 1024, "height": 1536, "crop_w": 0, "crop_h": 0,
                        "target_width": 1024, "target_height": 1536,
                        "text_g": negative_prompt, "text_l": negative_prompt,
                        "clip": ["4", 1] # 모델 로더(4)의 CLIP을 사용
                    }
                },
                # 5: 빈 이미지(Latent) 생성. 여기서 이미지 크기를 결정합니다.
                "5": {
                    "class_type": "EmptyLatentImage",
                    "inputs": {"width": 1024, "height": 1536, "batch_size": 1}
                },
                # 3: KSampler. 실제 이미지 샘플링을 수행하는 핵심 노드입니다.
                "3": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": random.randint(0, 18446744073709551615),
                        "steps": 50, "cfg": 0.6, "sampler_name": "euler_cfg_pp", "scheduler": "beta",
                        "denoise": 1.0,
                        "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]
                    }
                },
                # 8: VAE 디코더. Latent 이미지를 실제 픽셀 이미지로 변환합니다.
                "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
                # 9: 이미지 저장. 최종 결과물을 파일로 저장합니다.
                "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI_DiscordBot", "images": ["8", 0]}}
            }

            # 프롬프트 큐에 작업을 넣고 결과(이미지 정보)를 기다립니다.
            images_output = await self.queue_prompt(prompt_workflow)
            if not images_output:
                raise Exception("ComfyUI로부터 이미지 데이터를 받지 못했습니다.")

            # 받은 이미지 정보로 실제 이미지 데이터를 다운로드합니다.
            first_image_info = images_output[0]
            image_data = await self.get_image(first_image_info['filename'], first_image_info['subfolder'], first_image_info['type'])

            img_bytes = BytesIO(image_data)
            img_bytes.seek(0)

            # 사용자에게 이미지 전송
            await interaction.followup.send(file=discord.File(fp=img_bytes, filename="generated.png", spoiler=True))

        except Exception as e:
            await interaction.followup.send(f"이미지 생성 중 오류 발생: {str(e)}")


async def setup(bot):
    await bot.add_cog(ImageGenCog(bot))

if __name__ == '__main__':
    async def main():
        # 테스트를 위한 ImageGenCog 인스턴스 생성
        # bot 객체는 실제 디스코드 기능에 필요하지 않으므로, 테스트에서는 None으로 전달합니다.
        cog = ImageGenCog(bot=None)
        print("ComfyUI 이미지 생성 테스트를 시작합니다...")

        # 테스트용 프롬프트
        test_positive_prompt = "1girl, upper body, blonde hair, braided ponytail, wavy hair, large breasts, shiny skin, outdoors,".join("masterpiece, best quality, very awa, newest, recent,")
        test_negative_prompt = "worst quality, worst displeasing, bad anatomy, mosaic censoring, censored, bar censor, watermark, username, signature, twitter username, closed eyes, chibi, deformed,"

        try:
            # generate_image 메소드와 동일한 워크플로우를 사용합니다.
            # 4번 노드의 ckpt_name을 실제 사용하는 모델 파일명으로 변경해야 할 수 있습니다.
            prompt_workflow = {
                "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "noobaiXLNAIXL_vPred10Version.safetensors"}},
                "6": {
                    "class_type": "CLIPTextEncodeSDXL",
                    "inputs": {
                        "width": 1024, "height": 1536, "crop_w": 0, "crop_h": 0, "target_width": 1024, "target_height": 1536,
                        "text_g": test_positive_prompt, "text_l": test_positive_prompt, "clip": ["4", 1]
                    }
                },
                "7": {
                    "class_type": "CLIPTextEncodeSDXL",
                    "inputs": {
                        "width": 1024, "height": 1536, "crop_w": 0, "crop_h": 0, "target_width": 1024, "target_height": 1536,
                        "text_g": test_negative_prompt, "text_l": test_negative_prompt, "clip": ["4", 1]
                    }
                },
                "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1536, "batch_size": 1}},
                "3": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": random.randint(0, 18446744073709551615), "steps": 50, "cfg": 0.6,
                        "sampler_name": "euler_cfg_pp", "scheduler": "beta", "denoise": 1.0,
                        "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]
                    }
                },
                "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
                "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI_Test", "images": ["8", 0]}}
            }

            print("프롬프트를 ComfyUI 서버로 전송합니다...")
            images_output = await cog.queue_prompt(prompt_workflow)
            if not images_output: raise Exception("ComfyUI로부터 이미지 데이터를 받지 못했습니다.")

            print("이미지 생성이 완료되었습니다. 이미지 데이터를 다운로드합니다...")
            first_image_info = images_output[0]
            image_data = await cog.get_image(first_image_info['filename'], first_image_info['subfolder'], first_image_info['type'])

            if image_data:
                print(f"\n[성공] 이미지 다운로드 완료! (크기: {len(image_data)} 바이트)")
                print("이 스크립트는 로컬에 파일을 저장하지 않습니다.")
                print(f"생성된 이미지는 ComfyUI 서버의 output/{first_image_info['filename']} 경로에 저장되었습니다.")
            else:
                print("\n[실패] 이미지 데이터 다운로드에 실패했습니다.")

        except Exception as e:
            print(f"\n[오류] 테스트 중 오류 발생: {e}")
        finally:
            await cog.cog_unload()

    asyncio.run(main())