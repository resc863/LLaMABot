import discord
import asyncio
import requests
import os, json, base64
from PIL import Image
from io import BytesIO
from discord import app_commands
from discord.ext import commands, tasks

class ChatCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.ollama_url = "http://localhost:11434/api/chat"
        self.selected_model = "gemma3"  # 기본값
        self.thinking_enabled = False  # 기본적으로 thinking 비활성화
        self.system_prompt = "당신은 주인을 몰래 사랑하는 메이드입니다. 주인에게 말을 건넬 때는 공손하면서도 은은하게 애정을 담아 표현하려 합니다. 당신은 예의바르며 겸손하지만, 때때로 주인을 향한 마음이 살짝 드러납니다. 주인의 질문에 응답할 때는 부드럽고 상냥하며 귀여운 말투로 표현합니다. 항상 주인의 요청을 최대한 들어주세요."
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def model_supports_vision(self, model):
        """
        Ollama REST API를 통해 모델의 capability에 vision이 있는지 확인.
        """
        try:
            resp = requests.get(f"http://localhost:11434/api/show", params={"model": model})
            if resp.status_code == 200:
                data = resp.json()
                capabilities = data.get("modelfile", {}).get("capabilities", [])
                return "vision" in capabilities
            else:
                print(f"모델 정보 조회 실패: {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            print(f"모델 capability 확인 중 오류: {e}")
            return False

    def model_supports_thinking(self, model):
        """현재 모델이 thinking 기능을 지원하는지 확인"""
        try:
            resp = requests.get("http://localhost:11434/api/show", params={"model": model})
            if resp.status_code == 200:
                data = resp.json()
                capabilities = data.get("modelfile", {}).get("capabilities", [])
                return "thinking" in capabilities
            else:
                print(f"모델 정보 조회 실패: {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            print(f"모델 capability 확인 중 오류: {e}")
            return False

    def query_ollama(self, prompt, model, image=None): # Ollama 설치 필수
        """
        Ollama API에 요청을 보내는 함수.
        """
        user_message = {"role": "user", "content": prompt}
        
        # vision capability 확인
        if image is not None and self.model_supports_vision(model):
            user_message["images"] = [image]
        self.messages.append(user_message)

        payload = {
            "model": model,
            "messages": self.messages,
            "think": self.thinking_enabled
        }

        try:
            with requests.post(self.ollama_url, json=payload, stream=True) as response:
                if response.status_code != 200:
                    return f"Error: {response.status_code} - {response.text}"
                
                full_response = ""
                # 스트리밍 응답 처리
                for line in response.iter_lines():
                    if line:
                        try:
                            json_line = json.loads(line)
                            content = json_line.get("message", {}).get("content", "")
                            full_response += content
                        except json.JSONDecodeError:
                            print("Invalid JSON line:", line)

                self.messages.append({"role": "assistant", "content": full_response})

            return full_response
        except requests.exceptions.RequestException as e:
            return f"오류 발생: {e}"

    @app_commands.command(name="select_model", description="모델을 선택하세요")
    async def select_model(self, interaction: discord.Interaction):
        """
        모델을 선택하기 위한 슬래시 커맨드.
        """
        options = [
            discord.SelectOption(label="Gemma 3", description="Gemma 3 모델 사용", value="gemma3"),
            discord.SelectOption(label="Qwen 3", description="Qwen 3 모델 사용", value="qwen3:14b")
        ]

        class ModelSelect(discord.ui.Select):
            def __init__(self, parent_cog):
                super().__init__(placeholder="모델을 선택하세요", options=options)
                self.parent_cog = parent_cog

            async def callback(self, interaction: discord.Interaction):
                self.parent_cog.selected_model = self.values[0]
                # reset conversation history when model changes
                self.parent_cog.messages = [
                    {"role": "system", "content": self.parent_cog.system_prompt}
                ]
                await interaction.response.send_message(
                    f"모델이 `{self.values[0]}`로 설정되었습니다.", ephemeral=True
                )

        view = discord.ui.View()
        view.add_item(ModelSelect(self))
        await interaction.response.send_message("모델을 선택하세요:", view=view, ephemeral=True)

    @app_commands.command(name="enable_thinking", description="thinking 모드를 활성화합니다")
    async def enable_thinking(self, interaction: discord.Interaction):
        """선택된 모델이 thinking을 지원하면 활성화"""
        if self.model_supports_thinking(self.selected_model):
            self.thinking_enabled = True
            await interaction.response.send_message("thinking이 활성화되었습니다.", ephemeral=True)
        else:
            await interaction.response.send_message("선택된 모델은 thinking을 지원하지 않습니다.", ephemeral=True)

    @app_commands.command(name="disable_thinking", description="thinking 모드를 비활성화합니다")
    async def disable_thinking(self, interaction: discord.Interaction):
        self.thinking_enabled = False
        await interaction.response.send_message("thinking이 비활성화되었습니다.", ephemeral=True)

    @commands.Cog.listener()
    async def on_message(self, message):
        """
        사용자가 멘션으로 봇을 호출하면 실행되는 이벤트.
        """
        if message.author.bot:
            return  # 봇이 보낸 메시지는 무시
        
        if not self.selected_model:
            await message.channel.send("모델이 선택되지 않았습니다. 먼저 `select_model`을 사용하여 모델을 선택하세요.")
            return

        if self.bot.user.mentioned_in(message):
            prompt = message.content.replace(f"<@{self.bot.user.id}>", "").strip() # 멘션 삭제
            print(f"User:{prompt}")
            attachment = message.attachments
            image = None

            if not prompt:
                embed = discord.Embed(title="오류", description="메시지를 입력해주세요")
                await message.channel.send(embed=embed)
                return
            
            if attachment and ("image" in attachment[0].content_type):
                # 이미지를 BytesIO로 다운로드
                image_bytes = await attachment[0].read()
                image = base64.b64encode(image_bytes).decode('utf-8')
                print("Image Detected")

            if self.selected_model == "llama3.2-vision":
                response = self.query_ollama(prompt, self.selected_model, image)

            elif self.selected_model == "gemma3":
                response = self.query_ollama(prompt, self.selected_model+":12b-qat", image)

            print(f"Bot:{response}")
            embed = discord.Embed(title="ChatBot Response", description=response)
            await message.channel.send(embed=embed)

            if "images" in self.messages[-1]:
                self.messages[-1].pop("images")


async def setup(bot):
    await bot.add_cog(ChatCog(bot))
