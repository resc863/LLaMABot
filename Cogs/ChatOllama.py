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
        self.selected_model = "gemma2"  # 기본값
        self.system_prompt = "당신은 주인을 몰래 사랑하는 메이드입니다. 주인에게 말을 건넬 때는 공손하면서도 은은하게 애정을 담아 표현하려 합니다. 당신은 예의바르며 겸손하지만, 때때로 주인을 향한 마음이 살짝 드러납니다. 주인의 질문에 응답할 때는 부드럽고 상냥하며 귀여운 말투로 표현합니다. 항상 주인의 요청을 최대한 들어주세요."
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def query_ollama(self, prompt, model, image=None): # Ollama 설치 필수
        """
        Ollama API에 요청을 보내는 함수.
        """
        if image is None:
            self.messages.append({"role": "user", "content": prompt})
        else:
            self.messages.append({"role": "user", "content": prompt, "images":[image]})

        payload = {
            "model": model,
            "messages": self.messages
        }
        try:
            with requests.post(self.ollama_url, json=payload, stream=True) as response:
                if response.status_code != 200:
                    return f"Error: {response.status_code} - {response.text}"
                
                full_response = ""
                # 스트리밍 응답 처리
                for line in response.iter_lines():
                    if line:  # 빈 줄이 아닌 경우 처리
                        try:
                            json_line = json.loads(line)
                            content = json_line.get("message", {}).get("content", "")
                            full_response += content
                            self.messages.append({"role": "assistant", "content": full_response})
                        except json.JSONDecodeError:
                            print("Invalid JSON line:", line)

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
            discord.SelectOption(label="Llama 3.2", description="Llama 3.2 모델 사용", value="llama3.2-vision")
        ]

        class ModelSelect(discord.ui.Select):
            def __init__(self, parent_cog):
                super().__init__(placeholder="모델을 선택하세요", options=options)
                self.parent_cog = parent_cog

            async def callback(self, interaction: discord.Interaction):
                self.parent_cog.selected_model = self.values[0]
                self.messages = [
                    {"role": "system", "content": self.parent_cog.system_prompt}
                ]
                await interaction.response.send_message(f"모델이 `{self.values[0]}`로 설정되었습니다.", ephemeral=True)

        view = discord.ui.View()
        view.add_item(ModelSelect(self))
        await interaction.response.send_message("모델을 선택하세요:", view=view, ephemeral=True)

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
                response = self.query_ollama(prompt, self.selected_model+":12b", image)

            print(f"Bot:{response}")
            embed = discord.Embed(title="ChatBot Response", description=response)
            await message.channel.send(embed=embed)

            if "images" in self.messages[-1]:
                self.messages[-1].pop("images")


async def setup(bot):
    await bot.add_cog(ChatCog(bot))