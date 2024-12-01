import discord
import asyncio
import os, json
import torch
import transformers
import bitsandbytes
from PIL import Image
from io import BytesIO
from discord import app_commands
from discord.ext import commands, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, pipeline

class ChatTransformers(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.models = {
            "llama3.2": {
                "name": "LLaMA 3.2 Vision",
                "model_path": "model/llama3.2",
                "tokenizer": None,
                "model": None
            },
            "gemma2": {
                "name": "Gemma 2",
                "model_path": "model/gemma2",
                "tokenizer": None,
                "model": None
            },
        }
        self.current_model = None  # 현재 선택된 모델 이름
        self.system_prompt = "당신은 주인을 몰래 사랑하는 메이드입니다. 주인에게 말을 건넬 때는 공손하면서도 은은하게 애정을 담아 표현하려 합니다. 당신은 예의바르며 겸손하지만, 때때로 주인을 향한 마음이 살짝 드러납니다. 주인의 질문에 응답할 때는 부드럽고 상냥하며 귀여운 말투로 표현합니다. 항상 주인의 요청을 최대한 들어주세요."
        self.chat_history = [] # 채팅 내역 기억
        self.image = None # 멀티모달 모델 전용

    @app_commands.command(name="select_model", description="Select Chat Model")
    async def select_model(self, interaction: discord.Interaction):
        # Select 메뉴 구성
        options = [
            discord.SelectOption(label="LLaMA 3.2", description="Load the Llama 3.2 model. Everytime you upload photo, memory will reset", value="llama3.2"),
            discord.SelectOption(label="Gemma 2", description="Load the Gemma 2 model", value="gemma2"),
        ]

        class ModelSelect(discord.ui.Select):
            def __init__(self, parent_cog):
                super().__init__(placeholder="Select a model...", options=options)
                self.parent_cog = parent_cog

            async def callback(self, interaction: discord.Interaction):
                model_name = self.values[0]
                await interaction.response.defer(thinking=True)

                if model_name == "llama3.2": # llama 3는 한국어 성능이 빈약
                    self.parent_cog.chat_history = []
                    await interaction.followup.send(f"Progress: {model_name} is loading...", ephemeral=True)

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True, 
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16
                    ) #Multimodal feature requires more than 16GB VRAM
                    self.parent_cog.models[model_name]["model"] = MllamaForConditionalGeneration.from_pretrained( # For multimodal model
                        "model/llama3.2",
                        quantization_config=quantization_config,
                        device_map="auto",
                    )
                    self.parent_cog.models[model_name]["tokenizer"] = AutoProcessor.from_pretrained("model/llama3.2")
                    
                elif model_name == "gemma2":
                    self.parent_cog.chat_history = [
                        {
                            "role": "user", 
                            "content": self.parent_cog.system_prompt,
                        },
                        {
                            "role":"model",
                            "content":"알겠습니다."
                        }
                    ]

                    await interaction.followup.send(f"Progress: {model_name} is loading...", ephemeral=True) 
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # You can also try load_in_4bit
                    self.parent_cog.models[model_name]["model"] = pipeline("text-generation", "model/gemma2_9b", device_map="cuda", model_kwargs={"quantization_config": quantization_config}) # Use default text-generation pipeline

                self.parent_cog.current_model = model_name
                print(f"Model `{self.parent_cog.models[model_name]['name']}` has been loaded!")
                await interaction.followup.send(content=f"Model `{self.parent_cog.models[model_name]['name']}` has been loaded!")

        view = discord.ui.View()
        view.add_item(ModelSelect(self))
        await interaction.response.send_message("모델을 선택하세요:", view=view, ephemeral=True)

    def chat_llama(self, image=None):
        model_info = self.models[self.current_model]
        processor = model_info["tokenizer"] # 멀티모달 전처리
        model = model_info["model"]
        input_text = processor.apply_chat_template(self.chat_history, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=512)
        response = processor.decode(output[0][len(inputs["input_ids"][0]):-1])
        self.chat_history.append({
            "role":"assistant",
            "content":response
        })
        return response

    def chat_gemma(self):
        model_info = self.models[self.current_model]
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        response = model(self.chat_history, max_new_tokens=512)
        self.chat_history = response[0]['generated_text']
        return self.chat_history[-1]['content']

    @commands.Cog.listener()
    async def on_message(self, message):
        """
        사용자가 멘션으로 봇을 호출하면 실행되는 이벤트.
        """
        if message.author.bot:
            return  # 봇이 보낸 메시지는 무시
        
        if not self.current_model:
            await message.channel.send("모델이 선택되지 않았습니다. 먼저 `select_model`을 사용하여 모델을 선택하세요.")
            return

        if self.bot.user.mentioned_in(message):
            prompt = message.content.replace(f"<@{self.bot.user.id}>", "").strip() # 멘션 삭제
            print(f"User:{prompt}")
            attachment = message.attachments

            if not prompt:
                embed = discord.Embed(title="오류", description="메시지를 입력해주세요")
                await message.channel.send(embed=embed)
                return
            
            if self.current_model == "llama3.2":
                if attachment and ("image" in attachment[0].content_type):
                    # 이미지를 BytesIO로 다운로드
                    image_bytes = await attachment[0].read()
                    image = Image.open(BytesIO(image_bytes))
                    print("Image Detected")
                    self.chat_history.append({ # Only 1 image for entire history
                        "role":"user",
                        "content":[{"type": "image"},
                        {"type":"text", "text":prompt}]
                    })
                else:
                    self.chat_history.append({
                        "role":"user",
                        "content":prompt
                    })
                response = self.chat_llama(image)

                for content in self.chat_history[-1]:
                    if type(content) is not str and content["type"] == "image":
                        self.chat_history[-1]["content"].remove({"type": "image"})
                

            elif self.current_model == "gemma2":
                self.chat_history.append({
                    "role":"user",
                    "content":prompt
                })
                response = self.chat_gemma()

            print(f"Bot:{response}")
            embed = discord.Embed(title="ChatBot Response", description=response)
            await message.channel.send(embed=embed)
            
async def setup(bot):
    await bot.add_cog(ChatTransformers(bot))
    await bot.tree.sync()