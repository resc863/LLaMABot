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

    if not (await bot.is_owner(interaction.user)):
        ans = discord.Embed(
            title="Access Denied",
            description="You don't have permission for it.",
            color=0xcceeff,
        )
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

bot.run(token)