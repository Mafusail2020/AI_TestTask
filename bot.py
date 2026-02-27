import asyncio
import logging
import os
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile
from aiogram.filters import CommandStart

from config import config
from generate import build_workflow as build_generate_workflow
from analyze import build_analyze_workflow

bot = Bot(config.TOKEN)
dp = Dispatcher()

def get_main_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Згенерувати датасет", callback_data="generate_dataset")],
            [InlineKeyboardButton(text="🔍 Аналізувати датасет", callback_data="analyze_dataset")]
        ]
    )

@dp.message(CommandStart())
async def cmd_start(message: Message):
    """
    Welcoming message with inline buttons.
    """
    welcome_text = (
        "👋 Привіт, я невеликий бот зроблений для AITestTask!\n\n"
        "Моє завдання це просто трохи візуалізувати інтерфейс для юзера, тому тут лише дві кнопки.\n\n"
        "Будь ласка, вибери що хочеш зробити:"
    )
    await message.answer(welcome_text, reply_markup=get_main_keyboard())

@dp.callback_query(F.data == "generate_dataset")
async def process_generate(callback: CallbackQuery):
    """
    Handles the generation process.
    """
    await callback.answer("Генеруємо датасет...")
    status_msg = await callback.message.answer("⏳ Генерацію почато... Це може зайняти якийсь час.")
    
    try:
        app = build_generate_workflow()
        initial_state = {
            "num_dialogues": 5
        }
        
        await asyncio.to_thread(app.invoke, initial_state)
        
        file_path = "data/raw_dialogues.json"
        if os.path.exists(file_path):
            file = FSInputFile(file_path)
            await callback.message.answer_document(file, caption="✅ Датасет згенеровано! Прошу")
        else:
            await callback.message.answer("❌ Помилка: Шляху не існує.")
            
    except Exception as e:
        await callback.message.answer(f"❌ Сталася помилка під час генерації:\n{e}")
    finally:
        await status_msg.delete()

@dp.callback_query(F.data == "analyze_dataset")
async def process_analyze(callback: CallbackQuery):
    """
    Handles the analysis process.
    """
    await callback.answer("Аналізуємо датасет...")
    status_msg = await callback.message.answer("🔍 Аналіз почато... Чейкай.")
    
    try:
        app = build_analyze_workflow()
        initial_state = {
            "input_file": "data/raw_dialogues.json"
        }
        
        await asyncio.to_thread(app.invoke, initial_state)
        
        file_path = "data/analyzed_dialogues.json"
        if os.path.exists(file_path):
            file = FSInputFile(file_path)
            await callback.message.answer_document(file, caption="✅ Аналіз завершено!")
        else:
            await callback.message.answer("❌ Помилка: Шляху не існує.")
            
    except FileNotFoundError:
        await callback.message.answer("❌ Помилка: Спочатку згенеруйте датасет!")
    except Exception as e:
        await callback.message.answer(f"❌ Сталася помилка під час аналізу:\n{e}")
    finally:
        await status_msg.delete()

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        print("Starting bot...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped")