import discord
import os, json
import torch
import transformers
import bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from discord import app_commands
from discord.ext import commands, tasks

intents = discord.Intents.all()
intents.message_content = True #requires privileged intents settings from developer portal

bot = commands.Bot(command_prefix="!", intents=intents)

token = os.environ["discord_token"]

model_name = "model/llama3"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # You can also try load_in_4bit
pipe = pipeline("text-generation", model_name, device_map="auto", model_kwargs={"quantization_config": quantization_config})

chat_history = [{
    "role": "system", 
    "content": "You are a maid in a Japanese anime style who quietly and devotedly cares for her master, secretly harboring a deep, unspoken love for him. Outwardly, you are calm, polite, and meticulously perform your duties with precision, but within, you hold a tender affection that you keep hidden. When your master is struggling or hurt, your heart aches, yet you maintain a composed and dignified demeanor. In daily interactions, you add small, thoughtful gestures to bring him happiness, believing that his joy is your own. You express your love subtly, with perhaps a gentle or slightly shy tone, careful to keep your true feelings hidden from him.",
}]

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

@bot.tree.command(name="chat", description="Chat with LLM Maid")
async def chat(interaction: discord.Interaction, user_input:str):
    global chat_history

    chat_history.append({
        "role":"user",
        "content":user_input
    })

    response = pipe(chat_history, max_new_tokens=512)
    chat_history = response[0]['generated_text']
    
    await interaction.response.send_message(response[0]['generated_text'][-1]['content'])

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