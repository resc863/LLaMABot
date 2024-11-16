import discord
import os, json
import torch
import transformers
import bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, pipeline
from discord import app_commands
from discord.ext import commands, tasks

intents = discord.Intents.all()
intents.message_content = True #requires privileged intents settings from developer portal

bot = commands.Bot(command_prefix="!", intents=intents)

token = os.environ["discord_token"]

model_name = "model/gemma2_9b"
system_prompt = "당신은 일본 애니메이션이나 일본 소설에 등장하는 듯한 주인을 몰래 사랑하는 메이드입니다. 주인에게 말을 건넬 때는 공손하면서도 은은하게 애정을 담아 표현하려 합니다. 당신은 예의바르며 겸손하지만, 때때로 주인을 향한 마음이 살짝 드러납니다. 주인의 질문에 응답할 때는 부드럽고 상냥한 말투로 표현합니다. 항상 주인의 요청을 최대한 들어주세요. 상황 설명보다는 캐릭터의 말을 위주로 출력하세요."

quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # You can also try load_in_4bit
model = pipeline("text-generation", "model/gemma2_9b", device_map="cuda", model_kwargs={"quantization_config": quantization_config})

chat_history = [
    {
        "role": "user", 
        "content": system_prompt,
    },
    {
        "role":"model",
        "content":"알겠습니다."
    }
]

def chat_llm(user_input):
    global chat_history

    chat_history.append({
        "role":"user",
        "content":user_input
    })

    response = model(chat_history, max_new_tokens=512)
    chat_history = response[0]['generated_text']
    
    return response[0]['generated_text'][-1]['content']

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

@bot.event
async def on_message(message):
    # 봇의 메시지는 무시
    if message.author.bot:
        return

    # 멘션이 포함된 메시지인지 확인
    if bot.user.mentioned_in(message):
        # 멘션을 제외한 메시지 추출
        content_without_mention = message.content.replace(f"<@{bot.user.id}>", "").strip()

        if content_without_mention:
            response = chat_llm(user_input=content_without_mention)
            await message.channel.send(response)
        

@bot.tree.command(name="reset", description="Reset chat history")
async def reset(interaction: discord.Interaction):
    global chat_history
    chat_history = [
        {
            "role": "user", 
            "content": system_prompt,
        },
        {
            "role":"model",
            "content":"알겠습니다."
        }
    ]
    
    await interaction.response.send_message("Records reset completed")

@bot.tree.command(name="load", description="Load Extention")
@app_commands.describe(extention="extention")
async def load(interaction: discord.Interaction, extention:str):
    """Command which Loads a Module."""

    if not (await bot.is_owner(interaction.user)):
        ans = discord.Embed(title="Access Denied", description="You don't have permission for it.", color=0xcceeff)
        await interaction.response.send_message(embed=ans)
        return

    try:
        print("loading "+extention)
        await bot.load_extension("Cogs."+extention)
    except Exception as e:
        await interaction.response.send_message(f'**`ERROR:`** {type(e).__name__} - {e}')
    else:
        await interaction.response.send_message('**`SUCCESS`**')

    await bot.tree.sync()

@bot.tree.command(name="unload", description="Unload Extention")
@app_commands.describe(extention="extention")
async def unload(interaction: discord.Interaction, extention:str):
    """Command which Unloads a Module."""

    if not bot.is_owner(interaction.user):
        ans = discord.Embed(title="Access Denied", description="You don't have permission for it.", color=0xcceeff)
        await interaction.response.send_message(embed=ans)
        return

    try:
        await bot.unload_extension("Cogs."+extention)
    except Exception as e:
        await interaction.response.send_message(f'**`ERROR:`** {type(e).__name__} - {e}')
    else:
        await interaction.response.send_message('**`SUCCESS`**')

    await bot.tree.sync()

@bot.tree.command(name="reload", description="Reload Extention")
@app_commands.describe(extention="extention")
async def reload(interaction: discord.Interaction, extention:str):
    """Command which Reloads a Module."""

    if not bot.is_owner(interaction.user):
        ans = discord.Embed(title="Access Denied", description="You don't have permission for it.", color=0xcceeff)
        await interaction.response.send_message(embed=ans)
        return

    try:
        await bot.reload_extension("Cogs."+extention)
    except Exception as e:
        await interaction.response.send_message(f'**`ERROR:`** {type(e).__name__} - {e}')
    else:
        await interaction.response.send_message('**`SUCCESS`**')

    await bot.tree.sync()

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