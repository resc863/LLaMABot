import discord
import asyncio
import aiohttp
import os, json, base64
from PIL import Image
from io import BytesIO
from discord import app_commands
from discord.ext import commands, tasks

class ChatGemini(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.session = aiohttp.ClientSession()
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        self.api_key = os.getenv("GEMINI_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_KEY environment variable is not set.")

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
            self.user_states[user_id] = {
                "persona_key": self.default_persona_key,
                "messages": [],
            }
        return self.user_states[user_id]

    async def query_gemini(self, prompt, messages, persona, image=None):
        """
        Gemini API에 요청을 보내는 함수.
        """
        headers = {
            "Content-Type": "application/json"
        }

        # 사용자 메시지 추가
        user_parts = [{"text": prompt}]
        if image:
            user_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg", # Discord에서 받은 이미지의 MIME 타입에 따라 변경될 수 있음
                    "data": image
                }
            })
        
        # 대화 기록에 사용자 메시지 추가
        messages.append({"role": "user", "parts": user_parts})

        payload = {
            "contents": messages,
            "system_instruction": {
                "parts": [{"text": persona}]
            }
        }

        api_url = f"{self.api_url}?key={self.api_key}"

        full_response = ""
        try:
            async with self.session.post(api_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                if "candidates" in data and data["candidates"]:
                    for part in data["candidates"][0]["content"]["parts"]:
                        if "text" in part:
                            full_response += part["text"]
                else:
                    # 오류 응답 처리 개선
                    error_message = data.get("error", {}).get("message", "Unknown error")
                    print(f"Gemini API 응답 오류: {error_message}")
                    # 대화 기록에서 마지막 사용자 메시지 제거 (실패했으므로)
                    messages.pop()
                    return f"Gemini API 오류: {error_message}"

            # 대화 기록에 모델 응답 추가
            messages.append({"role": "model", "parts": [{"text": full_response}]})
            return full_response
        except aiohttp.ClientConnectionError:
            messages.pop()
            return "Gemini 서버에 연결할 수 없습니다. API 키와 네트워크 연결을 확인해주세요."
        except aiohttp.ClientError as e:
            messages.pop()
            return f"Gemini API 요청 중 오류가 발생했습니다: {e}"
        except json.JSONDecodeError:
            messages.pop()
            return "Gemini API 응답을 디코딩하는 중 오류가 발생했습니다."

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
        state["messages"] = [] # 메시지 기록 초기화
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
        state["messages"] = [] # 메시지 기록 초기화
        await interaction.response.send_message("대화 기록이 초기화되었습니다.", ephemeral=True)

    @commands.Cog.listener()
    async def on_message(self, message):
        """
        사용자가 멘션으로 봇을 호출하면 실행되는 이벤트.
        """
        if message.author.bot:
            return  # 봇이 보낸 메시지는 무시
        
        state = self.get_user_state(message.author.id)

        if self.bot.user.mentioned_in(message):
            prompt = message.content.replace(f"<@{self.bot.user.id}>", "").strip() # 멘션 삭제
            print(f"User:{prompt}")
            attachment = message.attachments
            image = None
            response = None

            if not prompt and not attachment:
                embed = discord.Embed(title="오류", description="메시지를 입력하거나 이미지를 첨부해주세요.")
                await message.channel.send(embed=embed)
                return
            
            try:
                if attachment and ("image" in attachment[0].content_type):
                    # 이미지를 BytesIO로 다운로드하여 base64 인코딩
                    image_bytes = await attachment[0].read()
                    image = base64.b64encode(image_bytes).decode('utf-8')
                    print("Image Detected")
            except discord.HTTPException as e:
                print(f"첨부파일 다운로드 실패: {e}")
                await message.channel.send("첨부파일을 처리하는 중 오류가 발생했습니다.")
                return

            # Gemini API 호출
            persona_key = state.get("persona_key", self.default_persona_key)
            persona_text = self.personas[persona_key]
            response = await self.query_gemini(prompt, state["messages"], persona_text, image)
            
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
    await bot.add_cog(ChatGemini(bot))