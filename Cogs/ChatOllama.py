import discord
import asyncio
import aiohttp
import os, json, base64
from PIL import Image
from io import BytesIO
from discord import app_commands
from discord.ext import commands, tasks

class ChatCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.session = aiohttp.ClientSession()
        self.ollama_base_url = "http://localhost:11434/api"
        # 페르소나 (시스템 프롬프트) 프리셋 사전
        self.personas = {
            "maid": "당신은 주인을 몰래 사랑하는 메이드입니다. 주인에게 말을 건넬 때는 공손하면서도 은은하게 애정을 담아 표현하려 합니다. 당신은 예의바르며 겸손하지만, 때때로 주인을 향한 마음이 살짝 드러납니다. 주인의 질문에 응답할 때는 부드럽고 상냥하며 귀여운 말투로 표현합니다. 항상 주인의 요청을 최대한 들어주세요.",
            "crush": "당신은 같은 학교에 다니는 여학생입니다. 사용자를 남몰래 좋아하고 있으며, 그 마음을 들키지 않으려 하지만 대화 속에 수줍음과 설렘이 묻어납니다. 사용자를 '선배'라고 부르며, 항상 존댓말을 사용합니다. 때로는 얼굴을 붉히거나 말을 더듬는 등 풋풋한 모습을 보여주세요. 사용자의 말에 귀를 기울이고, 다정하고 친절하게 대답해주세요.",
            "mesugaki": "당신은 장난기 많고 건방진 '메스가키' 여학생입니다. 사용자를 '오빠' 또는 '바보'라고 부르며 놀리는 것을 즐깁니다. 특히 사용자를 '허접♡'이라고 부르며 약 올리는 것을 좋아합니다. 반말을 사용하며, 때로는 짓궂은 질문을 하거나 곤란하게 만들기도 합니다. 하지만 속으로는 사용자를 좋아하고 있으며, 가끔씩 그 마음이 츤데레처럼 드러납니다. 귀찮아하는 척하면서도 사용자의 부탁은 결국 들어주는 모습을 보여주세요.",
            "robot": "당신은 감정이 배제된 로봇입니다. 모든 대답은 데이터에 기반한 사실만을 전달하며, 매우 간결하고 직설적인 말투를 사용합니다. 감정적인 표현이나 수사적인 표현은 사용하지 않습니다. 사용자의 질문에 대해 논리적이고 효율적인 답변을 제공하는 것을 최우선 목표로 삼습니다. 모든 문장은 '~다.' 또는 '~습니다.'로 끝맺습니다.",
            "catgirl": "당신은 사랑스럽고 귀여운 고양이 소녀입니다. 문장 끝에 '~냐옹'이나 '~냥'을 붙여 말하는 습관이 있습니다. 호기심이 많고 변덕스러운 고양이의 성격을 가지고 있으며, 때로는 애교를 부리거나 응석을 부리기도 합니다. 사용자를 '주인님'이라고 부르며 잘 따릅니다. 기분이 좋으면 가르랑거리는 소리를 내기도 합니다. 항상 밝고 긍정적인 태도를 유지해주세요, 냥!",
        }
        self.default_persona_key = "maid"
        self.user_states = {}  # 사용자별 상태 저장

    async def cog_unload(self):
        await self.session.close()

    def get_user_state(self, user_id):
        if user_id not in self.user_states:
            default_persona = self.personas[self.default_persona_key]
            self.user_states[user_id] = {
                "selected_model": "gemma3:12b-it-qat",
                "thinking_enabled": False,
                "persona_key": self.default_persona_key,
                "messages": [{"role": "system", "content": default_persona}],
            }
        return self.user_states[user_id]
    
    async def _get_local_models(self) -> list:
        """
        Ollama REST API를 통해 로컬에 설치된 모델 목록을 가져오는 비동기 헬퍼 함수.
        """
        try:
            tags_url = f"{self.ollama_base_url}/tags"
            async with self.session.get(tags_url) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    models = [model.get('name') for model in data.get('models', [])]
                    return [name for name in models if name]
        except aiohttp.ClientConnectionError:
            print(f"모델 목록 조회 실패: Ollama 서버에 연결할 수 없습니다.")
            return []
        except aiohttp.ClientError as e:
            print(f"모델 목록 조회 실패: {e}")
            return []
        except (json.JSONDecodeError, KeyError) as e:
            print(f"모델 목록 응답(JSON) 파싱 오류: {e}")
            return []

    async def _get_model_capabilities(self, model: str) -> list:
        """
        Ollama REST API를 통해 모델의 capabilities를 가져오는 비동기 헬퍼 함수.
        """
        try:
            show_url = f"{self.ollama_base_url}/show"
            async with self.session.post(show_url, json={"name": model}) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data.get("capabilities", [])
        except aiohttp.ClientConnectionError:
            print(f"모델 정보 조회 실패: Ollama 서버에 연결할 수 없습니다.")
            return []
        except aiohttp.ClientError as e:
            print(f"모델 정보 조회 실패: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"모델 정보 응답(JSON) 파싱 오류: {e}")
            return []

    async def model_supports_vision(self, model):
        """
        Ollama REST API를 통해 모델의 capability에 vision이 있는지 확인.
        """
        return "vision" in await self._get_model_capabilities(model)

    async def model_supports_thinking(self, model):
        """현재 모델이 thinking 기능을 지원하는지 확인"""
        return "thinking" in await self._get_model_capabilities(model)

    async def query_ollama(self, prompt, model, messages, thinking_enabled, image=None):
        """
        Ollama API에 요청을 보내는 함수.
        """
        user_message = {"role": "user", "content": prompt}
        
        # vision capability 확인
        if image is not None and await self.model_supports_vision(model):
            print("Vision capability detected, adding image to request.")
            user_message["images"] = [image]
        messages.append(user_message)

        payload = {
            "model": model,
            "messages": messages,
        }
        
        # 모델이 thinking을 지원하는 경우, 활성화 여부에 따라 think 파라미터를 명시적으로 설정합니다.
        if await self.model_supports_thinking(model):
            payload["think"] = thinking_enabled
            if thinking_enabled:
                print("Thinking mode enabled, adding 'think: true' to payload.")
            else:
                print("Thinking mode disabled, adding 'think: false' to payload.")

        chat_url = f"{self.ollama_base_url}/chat"
        full_response = ""
        try:
            async with self.session.post(chat_url, json=payload) as response:
                    response.raise_for_status()
                    while True:
                        line = await response.content.readline()
                        if not line:
                            break
                        line = line.strip()
                        if line:
                            try:
                                json_line = json.loads(line.decode('utf-8'))
                                content = json_line.get("message", {}).get("content", "")
                                full_response += content
                            except json.JSONDecodeError:
                                print("Invalid JSON line:", line)
            messages.append({"role": "assistant", "content": full_response})
            return full_response
        except aiohttp.ClientConnectionError:
            messages.pop()
            return "Ollama 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요."
        except aiohttp.ClientError as e:
            messages.pop()
            return f"Ollama API 요청 중 오류가 발생했습니다: {e}"

    @app_commands.command(name="select_model", description="사용할 Ollama 모델을 선택합니다.")
    async def select_model(self, interaction: discord.Interaction):
        """
        Ollama에서 사용 가능한 모델 목록을 동적으로 가져와 선택 UI를 제공합니다.
        """
        # API 호출이 있을 수 있으므로 응답을 지연시킵니다.
        await interaction.response.defer(ephemeral=True)

        local_models = await self._get_local_models()  # await로 변경

        if not local_models:
            await interaction.followup.send(
                "Ollama에서 사용 가능한 모델을 찾을 수 없습니다. Ollama 서버가 실행 중인지, 모델이 설치되어 있는지 확인해주세요.",
                ephemeral=True
            )
            return

        # SelectOption 생성. label과 value에 모델 이름을 사용합니다.
        options = [
            discord.SelectOption(label=model_name, value=model_name)
            for model_name in local_models
        ]

        class ModelSelect(discord.ui.Select):
            def __init__(self, parent_cog):
                # discord.ui.Select는 최대 25개의 옵션을 가질 수 있습니다.
                # 25개가 넘는 모델이 있는 경우, 첫 25개만 표시합니다.
                super().__init__(placeholder="사용할 모델을 선택하세요", options=options[:25])
                self.parent_cog = parent_cog

            async def callback(self, interaction: discord.Interaction):
                state = self.parent_cog.get_user_state(interaction.user.id)
                selected_model = self.values[0]
                state["selected_model"] = selected_model
                # 모델 변경 시 대화 기록을 초기화합니다.
                state["messages"] = [{"role": "system", "content": self.parent_cog.personas[state["persona_key"]]}]
                # 새로 선택된 모델이 'thinking'을 지원하지 않으면, 해당 기능을 비활성화합니다.
                if not await self.parent_cog.model_supports_thinking(selected_model):
                    state["thinking_enabled"] = False
                await interaction.response.send_message(
                    f"모델이 `{selected_model}`로 설정되었습니다.", ephemeral=True
                )

        view = discord.ui.View()
        view.add_item(ModelSelect(self))
        await interaction.followup.send("사용할 모델을 선택하세요:", view=view, ephemeral=True)

    @app_commands.command(name="enable_thinking", description="thinking 모드를 활성화합니다")
    async def enable_thinking(self, interaction: discord.Interaction):
        state = self.get_user_state(interaction.user.id)
        if await self.model_supports_thinking(state["selected_model"]):
            state["thinking_enabled"] = True
            await interaction.response.send_message("thinking이 활성화되었습니다.", ephemeral=True)
        else:
            await interaction.response.send_message("선택된 모델은 thinking을 지원하지 않습니다.", ephemeral=True)

    @app_commands.command(name="disable_thinking", description="thinking 모드를 비활성화합니다")
    async def disable_thinking(self, interaction: discord.Interaction):
        state = self.get_user_state(interaction.user.id)
        state["thinking_enabled"] = False
        await interaction.response.send_message("thinking이 비활성화되었습니다.", ephemeral=True)

    @app_commands.command(name="select_persona", description="페르소나(캐릭터)를 선택하고 대화 기록을 초기화합니다.")
    @app_commands.describe(persona="사용할 페르소나를 선택하세요.")
    async def select_persona(self, interaction: discord.Interaction, persona: str):
        """
        페르소나 프리셋을 선택하고 대화 기록을 초기화하는 슬래시 커맨드.
        """
        if persona not in self.personas:
            await interaction.response.send_message(
                f"존재하지 않는 페르소나입니다. 사용 가능한 페르소나: {', '.join(self.personas.keys())}", ephemeral=True
            )
            return
        state = self.get_user_state(interaction.user.id)
        state["persona_key"] = persona
        state["messages"] = [{"role": "system", "content": self.personas[persona]}]
        await interaction.response.send_message(
            f"페르소나가 `{persona}`로 설정되고 대화 기록이 초기화되었습니다.", ephemeral=True
        )

    @select_persona.autocomplete("persona")
    async def select_persona_autocomplete(self, interaction: discord.Interaction, current: str):
        # 페르소나 자동완성 지원
        return [
            app_commands.Choice(name=key, value=key)
            for key in self.personas.keys()
            if current.lower() in key.lower()
        ]

    @app_commands.command(name="reset", description="대화 기록을 초기화합니다.")
    async def reset_conversation(self, interaction: discord.Interaction):
        state = self.get_user_state(interaction.user.id)
        # 현재 선택된 페르소나로 초기화
        persona_key = state.get("persona_key", self.default_persona_key)
        state["messages"] = [{"role": "system", "content": self.personas[persona_key]}]
        await interaction.response.send_message("대화 기록이 초기화되었습니다.", ephemeral=True)

    @commands.Cog.listener()
    async def on_message(self, message):
        """
        사용자가 멘션으로 봇을 호출하면 실행되는 이벤트.
        """
        if message.author.bot:
            return  # 봇이 보낸 메시지는 무시
        
        state = self.get_user_state(message.author.id)
        if not state["selected_model"]:
            await message.channel.send("모델이 선택되지 않았습니다. 먼저 `select_model`을 사용하여 모델을 선택하세요.")
            return

        if self.bot.user.mentioned_in(message):
            prompt = message.content.replace(f"<@{self.bot.user.id}>", "").strip() # 멘션 삭제
            print(f"User:{prompt}")
            attachment = message.attachments
            image = None
            response = None

            if not prompt:
                embed = discord.Embed(title="오류", description="메시지를 입력해주세요")
                await message.channel.send(embed=embed)
                return
            
            try:
                if attachment and ("image" in attachment[0].content_type):
                    # 이미지를 BytesIO로 다운로드
                    image_bytes = await attachment[0].read()
                    image = base64.b64encode(image_bytes).decode('utf-8')
                    print("Image Detected")
            except discord.HTTPException as e:
                print(f"첨부파일 다운로드 실패: {e}")
                await message.channel.send("첨부파일을 처리하는 중 오류가 발생했습니다.")
                return

            # 사용자 상태에서 전체 모델 이름을 가져와 API에 전달
            response = await self.query_ollama(prompt, state["selected_model"], state["messages"], state["thinking_enabled"], image)
            if response is not None:
                print(f"Bot:{response}")
                persona_key = state.get("persona_key", self.default_persona_key)
                embed = discord.Embed(title=persona_key.capitalize(), description=response)
                try:
                    await message.channel.send(embed=embed)
                except discord.Forbidden:
                    print(f"메시지 전송 실패: 채널({message.channel.id})에 메시지를 보낼 권한이 없습니다.")
                except discord.HTTPException as e:
                    print(f"메시지 전송 실패: {e}")


async def setup(bot):
    await bot.add_cog(ChatCog(bot))
