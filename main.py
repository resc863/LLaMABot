import discord
import asyncio
import os, json
import torch
import transformers
import bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, pipeline
from PIL import Image
from io import BytesIO
from discord import app_commands
from discord.ext import commands, tasks

intents = discord.Intents.all()
intents.message_content = True #requires privileged intents settings from developer portal

bot = commands.Bot(command_prefix="!", intents=intents)
token = os.environ["discord_token"]

model = None
model_name = ""
processor = None
gemma_prompt = "당신은 일본 애니메이션이나 일본 소설에 등장하는 듯한 주인을 몰래 사랑하는 메이드입니다. 주인에게 말을 건넬 때는 공손하면서도 은은하게 애정을 담아 표현하려 합니다. 당신은 예의바르며 겸손하지만, 때때로 주인을 향한 마음이 살짝 드러납니다. 주인의 질문에 응답할 때는 부드럽고 상냥한 말투로 표현합니다. 항상 주인의 요청을 최대한 들어주세요. 상황 설명보다는 캐릭터의 말을 위주로 출력하세요."

def chat_llama(user_input, image, chat_history, model, processor):

    if image is not None:
        chat_history = [{
            "role":"user",
            "content":[{"type": "image"},
            {"type":"text", "text":user_input}]
        }]
    else:
        for chat in chat_history:
            for content in chat["content"]:
                if type(content) is not str and content["type"] == "image":
                    chat_history = []

        chat_history.append({
            "role":"user",
            "content":user_input
        })

    input_text = processor.apply_chat_template(chat_history, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=256)
    chat_history.append({
        "role":"assistant",
        "content":processor.decode(output[0][len(inputs["input_ids"][0]):-1])
    })
    
    return processor.decode(output[0][len(inputs["input_ids"][0]):-1]), chat_history

def chat_gemma(user_input, chat_history, model):

    chat_history.append({
        "role":"user",
        "content":user_input
    })

    response = model(chat_history, max_new_tokens=256)
    chat_history = response[0]['generated_text']
    
    return response[0]['generated_text'][-1]['content'], chat_history

@bot.event
async def on_ready():
    print("Logged in ")
    print(bot.user.name)
    print(bot.user.id)
    print("===============\n")

    await bot.tree.sync()

    for i in bot.guilds:
        print(i)
    await bot.change_presence(status=discord.Status.idle,
                              activity=discord.Game("Loading..."))
    
class ModelSelect(discord.ui.Select):
    def __init__(self):
        # Select 메뉴 구성
        options = [
            discord.SelectOption(label="Llama 3.2", description="Load the Llama 3.2 model. Everytime you upload photo, memory will reset", value="llama3.2"),
            discord.SelectOption(label="Gemma 2", description="Load the Gemma 2 model", value="gemma2"),
        ]
        super().__init__(placeholder="Select a model...", options=options)

    async def callback(self, interaction: discord.Interaction):
        global model
        global model_name
        global processor
        global chat_history

        model_name = self.values[0]  # 선택된 모델 이름 저장

        await interaction.response.defer()

        if model_name == "llama3.2":
            chat_history = []
            await interaction.followup.send(f"Progress: Model is loading...", ephemeral=True)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            ) #Multimodal feature requires more than 16GB VRAM
            model = MllamaForConditionalGeneration.from_pretrained(
                "model/llama3.2",
                quantization_config=quantization_config,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained("model/llama3.2")
            
        elif model_name == "gemma2":
            chat_history = [
                {
                    "role": "user", 
                    "content": gemma_prompt,
                },
                {
                    "role":"model",
                    "content":"알겠습니다."
                }
            ]
            await interaction.followup.send(f"Progress: Model is loading...", ephemeral=True)
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # You can also try load_in_4bit
            model = pipeline("text-generation", "model/gemma2_9b", device_map="cuda", model_kwargs={"quantization_config": quantization_config})

        await interaction.followup.send(f"Model `{model_name}` has been loaded!", ephemeral=True)

class ModelSelectView(discord.ui.View):
    def __init__(self):
        super().__init__()
        self.add_item(ModelSelect())

@bot.tree.command(name="load", description="load chat model")
async def load(interaction: discord.Interaction):
    """모델 선택 메뉴를 표시합니다."""
    view = ModelSelectView()
    await interaction.response.send_message("Please select a model from the menu below:", view=view)

@bot.event
async def on_message(message):
    # 봇의 메시지는 무시
    if message.author.bot:
        return

    # 멘션이 포함된 메시지인지 확인
    if bot.user.mentioned_in(message):
        # 멘션을 제외한 메시지 추출
        content_without_mention = message.content.replace(f"<@{bot.user.id}>", "").strip()
        attachment = message.attachments
        image = None

        if content_without_mention:
            global model_name
            global chat_history
            global model

            if model_name == "llama3.2":
                global processor

                if attachment and ("image" in attachment[0].content_type):
                    # 이미지를 BytesIO로 다운로드
                    image_bytes = await attachment[0].read()
                    image = Image.open(BytesIO(image_bytes))
                    print("Image Detected")
                else:
                    image = None
                
                response, chat_history = chat_llama(content_without_mention, image, chat_history, model, processor)
                await message.channel.send(response)

            elif model_name == "gemma2":
                response, chat_history = chat_gemma(content_without_mention, chat_history, model)
                await message.channel.send(response)

@bot.tree.command(name="reset", description="Reset chat history")
async def reset(interaction: discord.Interaction):
    global chat_history
    global model_name
    
    if model_name == "llama3.2":
        chat_history = []

    elif model_name == "gemma2":
        chat_history = [
            {
                "role": "user", 
                "content": gemma_prompt,
            },
            {
                "role":"model",
                "content":"알겠습니다."
            }
        ]
    
    await interaction.response.send_message("Records reset completed")

@bot.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send("반갑습니다 " + member.guild.name + "에 오신것을 환영합니다")

@bot.event
async def on_raw_reaction_add(payload):
    message = await bot.get_channel(payload.channel_id
                                    ).fetch_message(payload.message_id)
    user = payload.member
    print(payload.emoji)
    print(message.channel.id)
    print(user)


@bot.event
async def on_raw_reaction_remove(payload):
    channel = await bot.fetch_channel(payload.channel_id)
    guild = channel.guild

    user = await guild.fetch_member(payload.user_id)
    print(channel.id)
    message = await channel.fetch_message(payload.message_id)
    print(payload.emoji)
    print(guild.system_channel)


bot.run(token)