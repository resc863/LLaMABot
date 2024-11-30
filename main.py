import discord
import asyncio
import os, json
from discord import app_commands
from discord.ext import commands, tasks

intents = discord.Intents.all()
intents.message_content = True #requires privileged intents settings from developer portal

bot = commands.Bot(command_prefix="!", intents=intents)
token = os.environ["discord_token"]

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

@bot.tree.command(name="load", description="Load Extention")
@app_commands.describe(extention="extention")
async def load(interaction: discord.Interaction, extention:str):
    """Command which Loads a Module."""

    if not (await bot.is_owner(interaction.user)):
        ans = discord.Embed(title="Access Denied", description="You don't have permission for it.", color=0xcceeff)
        await interaction.response.send_message(embed=ans)
        return
    
    await interaction.response.defer()
    try:
        print("loading "+extention)
        await bot.load_extension("Cogs."+extention)
    except Exception as e:
        await interaction.followup.send(f'**`ERROR:`** {type(e).__name__} - {e}', ephemeral=True)
    else:
        await interaction.followup.send('**`SUCCESS`**', ephemeral=True)

    await bot.tree.sync()

@bot.tree.command(name="unload", description="Unload Extention")
@app_commands.describe(extention="extention")
async def unload(interaction: discord.Interaction, extention:str):
    """Command which Unloads a Module."""

    if not bot.is_owner(interaction.user):
        ans = discord.Embed(title="Access Denied", description="You don't have permission for it.", color=0xcceeff)
        await interaction.response.send_message(embed=ans)
        return

    await interaction.response.defer()
    try:
        await bot.unload_extension("Cogs."+extention)
    except Exception as e:
        await interaction.followup.send(f'**`ERROR:`** {type(e).__name__} - {e}', ephemeral=True)
    else:
        await interaction.followup.send('**`SUCCESS`**', ephemeral=True)

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