from types import NoneType
from typing import List, Union
import openai
import tiktoken
import re
from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import api_key
from app import api_timeout,api_timeout2,api_key,api_key2,base_url,base_url2,max_len_chunk,spromt,spromt2,debug


# discrete chunks to translate one chunk at a time
MAX_TOKENS_PER_CHUNK = max_len_chunk*2 # if text is more than this many tokens, we'll break it up into, bytes x2 to tokens
outline_text = ""

client = openai.OpenAI(
    api_key=api_key,  # This is the default and can be omitted
    base_url=base_url,
    timeout=api_timeout
)
client2 = openai.OpenAI(
    api_key=api_key2,  # This is the default and can be omitted
    base_url=base_url2,
    timeout=api_timeout2
)

def remove_tags(text):
    # Определяем шаблон для поиска тегов и текста между ними
    pattern = r'<SOURCE_TEXT>.*?</SOURCE_TEXT>|<FIRST_VERSION_TRANSLATION>.*?</FIRST_VERSION_TRANSLATION>|<EXPERT_SUGGESTIONS>.*?</EXPERT_SUGGESTIONS>|<TRANSLATION>.*?</TRANSLATION>|<TTEXT>|</TTEXT>|<OUTLINE>.*?</OUTLINE>|'
    # Используем re.sub для удаления найденных совпадений
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text


def check_and_print_tags(text):
    # Определяем шаблон для поиска тегов
    pattern = r'<SOURCE_TEXT>.*?</SOURCE_TEXT>|<FIRST_VERSION_TRANSLATION>.*?</FIRST_VERSION_TRANSLATION>|<EXPERT_SUGGESTIONS>.*?</EXPERT_SUGGESTIONS>|<TRANSLATION>.*?</TRANSLATION>|<TTEXT>|</TTEXT>|<OUTLINE>.*?</OUTLINE>|```xml.*?```'
    # Ищем все вхождения тегов
    matches = re.findall(pattern, text)

    return matches

def get_completion(
        prompt: str,
        system_message: str ,
        model: str = "gpt-4-turbo",
        temperature: float = 0.15,
        json_mode: bool = False,
) -> Union[str, dict]:
    """
        Generate a completion using the OpenAI API.

    Args:
        prompt (str): The user's prompt or query.
        system_message (str, optional): The system message to set the context for the assistant.
            Defaults to "You are a helpful assistant.".
        model (str, optional): The name of the OpenAI model to use for generating the completion.
            Defaults to "gpt-4-turbo".
        temperature (float, optional): The sampling temperature for controlling the randomness of the generated text.
            Defaults to 0.3.
        json_mode (bool, optional): Whether to return the response in JSON format.
            Defaults to False.

    Returns:
        Union[str, dict]: The generated completion.
            If json_mode is True, returns the complete API response as a dictionary.
            If json_mode is False, returns the generated text as a string.
    """
    if spromt is None:
        prompt =  f"{system_message}.{prompt}"
        system_message = None

    if json_mode:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

def get_completion2(
        prompt: str,
        system_message: str ,
        model: str = "gpt-4-turbo",
        temperature: float = 0.8,
        json_mode: bool = False,
) -> Union[str, dict]:
    """
        Generate a completion using the OpenAI API.

    Args:
        prompt (str): The user's prompt or query.
        system_message (str, optional): The system message to set the context for the assistant.
            Defaults to "You are a helpful assistant.".
        model (str, optional): The name of the OpenAI model to use for generating the completion.
            Defaults to "gpt-4-turbo".
        temperature (float, optional): The sampling temperature for controlling the randomness of the generated text.
            Defaults to 0.3.
        json_mode (bool, optional): Whether to return the response in JSON format.
            Defaults to False.

    Returns:
        Union[str, dict]: The generated completion.
            If json_mode is True, returns the complete API response as a dictionary.
            If json_mode is False, returns the generated text as a string.
    """
    if spromt2 is None:
        prompt = system_message.prompt
        system_message = None

    if json_mode:
        response = client2.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    else:
        response = client2.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

def one_chunk_initial_translation(
        source_lang: str, target_lang: str, source_text: str, style: str, outline_text: str,
) -> str:
    """
    Translate the entire text as one chunk using an LLM.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text (str): The text to be translated.

    Returns:
        str: The translated text.
    """
    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
    if style == 'xml':  # Исправлено '=' на '=='
       translation_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this XML text in tags <TTEXT> </TTEXT>. \
    1. Do NOT provide any explanations or text apart from the translation.  \n
    2. Don't add markdown.\n 3. Don't add any comments. \n 4. Xml tags must be unchanged and untranslated. \n
    5. Use proper names and data from the synopsis of the previous part of the text marked as an synopsis in tags <OUTLINE> </OUTLINE>.\n
    <OUTLINE>{outline_text}</OUTLINE> \n
       
    <TTEXT>{source_text}</TTEXT>
    """

    else :
        translation_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for text in tags <TTEXT> </TTEXT>  \
            1. Do NOT provide any explanations or text apart from the translation.  \n
            2. Don't add any comments. \n 3. Do not use any tags in result. \n 4. Use proper names and data from the synopsis of the previous part of the text marked as an synopsis in tags <OUTLINE> </OUTLINE>.
            <OUTLINE>{outline_text}</OUTLINE> \n
            
            <TTEXT>{source_text}</TTEXT>
            """

    translation = get_completion(translation_prompt, system_message=system_message)
    translation = remove_tags(translation)
    return translation


def one_chunk_referat(
        source_lang: str, target_lang: str, final_translation : str
) -> str:
    """
    Make the referat for chunk using an LLM.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text (str): The text to be translated.

    Returns:
        str: The translated text.
    """
    system_message = f"You are an expert linguist, proofreader,  specializing in translation from {source_lang} to {target_lang}."
#    if style == 'xml':  # Исправлено '=' на '=='
    translation_prompt = f"""Provide a synopsis in {target_lang} language for this chunk of text so that the subsequent chunk of text can be better understood. \n
        Note the following :
        \n1. List the names and gender of the characters used, as well as Proper Nouns and Common Nouns.
        \n2. The synopsis must be as concise as possible. 50 words as maximum.
        \n3. Do not use references and comments in the synopsis.
        \n4. The result should be an synopsis in text format only.\n

       text: {final_translation}

       synopsis in {target_lang}:
       """


    return get_completion2(translation_prompt, system_message=system_message)


def one_chunk_editor(target_lang: str,  source_text: str, style: str
) -> str:
    """
    Args:
        target_lang (str): The target language for translation.
        source_text (str): The text to be translated.

    Returns:
        str: The adopted text.
    """
    system_message = f"You are an expert linguist specializing in {target_lang} language and proofreading of fiction texts."

    if style == 'xml':  # Исправлено '=' на '=='
        translation_prompt = f"""This is a XML code with text that contains many errors, you need to correct them all. \n   
    Don't use markdown, the xml tags must be unchanged. Output improved text and nothing else. Don't add any comment. \n
    {source_text} """
    else:  # Отступ исправлен
        translation_prompt = f"""This is a text that contains many errors, you need to correct them all. \n   
            Output improved text and nothing else. Don't add any comment. \n
            {source_text} """

    editor_text = get_completion2(translation_prompt, system_message=system_message)

    return editor_text


def one_chunk_reflect_on_translation(
        source_lang: str,
        target_lang: str,
        source_text: str,
        translation_1: str,
        country: str = "",
) -> str:
    """
    Use an LLM to reflect on the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        country (str): Country specified for the target language.

    Returns:
        str: The LLM's reflection on the translation, providing constructive criticism and suggestions for improvement.
    """

    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    if country != "":
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
        The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

        The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <FIRST_VERSION_TRANSLATION></FIRST_VERSION_TRANSLATION>, are as follows:

        <SOURCE_TEXT>
        {source_text}
        </SOURCE_TEXT>

        <FIRST_VERSION_TRANSLATION>
        {translation_1}
        </FIRST_VERSION_TRANSLATION>

        When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
        1. accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
        2. fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
        3. style (by ensuring the translations reflect the style of the source text and take into account any cultural context),\n\
        4. terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

        Write a list of specific, helpful and constructive suggestions for improving the translation.
        Each suggestion should address one specific part of the translation.
        Output only the list of short suggestions and nothing else."""

    else:
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticisms and helpful suggestions to improve the translation. \

        The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <FIRST_VERSION_TRANSLATION></FIRST_VERSION_TRANSLATION>, are as follows:

        <SOURCE_TEXT>
        {source_text}
        </SOURCE_TEXT>

        <FIRST_VERSION_TRANSLATION>
        {translation_1}
        </FIRST_VERSION_TRANSLATION>

        When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
        1. accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
        2. fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
        3. style (by ensuring the translations reflect the style of the source text and take into account any cultural context),\n\
        4. terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

        Write a list of specific, helpful and constructive suggestions for improving the translation.
        Each suggestion should address one specific part of the translation.
        Output only the list of suggestions and nothing else."""

    reflection = get_completion2(reflection_prompt, system_message=system_message)

    if debug:
        ic(check_and_print_tags(reflection))
    reflection = remove_tags(reflection)
    return reflection


def one_chunk_improve_translation(
        source_lang: str,
        target_lang: str,
        source_text: str,
        translation_1: str,
        reflection: str,
        style: str,
) -> str:
    """
    Use the reflection to improve the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        reflection (str): Expert suggestions and constructive criticism for improving the translation.

    Returns:
        str: The improved translation based on the expert suggestions.
    """

    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."
    if style == 'xml':  # Исправлено '=' на '=='
        prompt = f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
        account a list of expert suggestions and constructive criticisms.

        The source text, the initial translation, and the expert linguist suggestions are delimited by tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
        as follows:

        <SOURCE_TEXT>
        {source_text}
        </SOURCE_TEXT>

        <TRANSLATION>
        {translation_1}
        </TRANSLATION>

        <EXPERT_SUGGESTIONS>
        {reflection}
        </EXPERT_SUGGESTIONS>

        Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

        1. accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
        2. fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
        3. style (by ensuring the translations reflect the style of the source text)
        4. terminology (inappropriate for context, inconsistent use), or
        5. The result must not contain any tags: <SOURCE_TEXT>,<FIRST_VERSION_TRANSLATION>,<EXPERT_SUGGESTIONS>,<TRANSLATION>,<TTEXT>,<OUTLINE>
        6. other errors.
        Don't use markdown, the xml tags should be unchanged. Output ONLY improved translation and nothing else. Don't add any comment. """
    else:
        prompt = f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
        account a list of expert suggestions and constructive criticisms.

        The source text, the initial translation, and the expert linguist suggestions are delimited by tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
        as follows:

        <SOURCE_TEXT>
        {source_text}
        </SOURCE_TEXT>

        <TRANSLATION>
        {translation_1}
        </TRANSLATION>

        <EXPERT_SUGGESTIONS>
        {reflection}
        </EXPERT_SUGGESTIONS>

        Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

        1. accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
        2. fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
        3. style (by ensuring the translations reflect the style of the source text)
        4 terminology (inappropriate for context, inconsistent use), or
        5. The result must not contain any tags: <SOURCE_TEXT>,<FIRST_VERSION_TRANSLATION>,<EXPERT_SUGGESTIONS>,<TRANSLATION>,<TTEXT>,<OUTLINE>
        6. other errors.
        
        Output ONLY improved translation and nothing else. Don't add any comment. """

    translation_2 = get_completion(prompt, system_message)
    #Проверка на коды и мусор, очистка

    ic(check_and_print_tags(translation_2))

    # Удаляем текст между тегами
    translation_2  = remove_tags(translation_2 )
    #print("Очищенный текст:", output_text)


    return translation_2


def one_chunk_translate_text(
        source_lang: str, target_lang: str, source_text: str, style: str, outline_text: str,country: str = "",
) -> str:
    """
    Translate a single chunk of text from the source language to the target language.

    This function performs a two-step translation process:
    1. Get an initial translation of the source text.
    2. Reflect on the initial translation and generate an improved translation.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The text to be translated.
        country (str): Country specified for the target language.
    Returns:
        str: The improved translation of the source text.
    """
#This code must be switched to 2 LLM  tr1(llm1)-reflect(llm1)-tr2(tuned to target lang llm)

    translation_1 = one_chunk_initial_translation(
        source_lang, target_lang, source_text, style, outline_text
    )
    if debug:
        ic((num_tokens_in_string(translation_1)),style, outline_text,translation_1)


    reflection = one_chunk_reflect_on_translation(
        source_lang, target_lang, source_text, translation_1, country
    )

    if debug:
        ic((num_tokens_in_string(reflection)),style,reflection)
    translation_2 = one_chunk_improve_translation(
        source_lang, target_lang, source_text, translation_1, reflection, style
    )

    if debug:
        ic((num_tokens_in_string(translation_2)),style,translation_2)

    final_translation = one_chunk_editor(target_lang,translation_2, style)

    outline_text = one_chunk_referat(
        source_lang, target_lang, final_translation,
    )
    if debug:
        ic((num_tokens_in_string(outline_text)),outline_text)


    if debug:
        ic((num_tokens_in_string(final_translation)),final_translation)

    return final_translation, outline_text

def num_tokens_in_string(
        input_str: str, encoding_name: str = "cl100k_base"
) -> int:
    """
    Calculate the number of tokens in a given string using a specified encoding.

    Args:
        str (str): The input string to be tokenized.
        encoding_name (str, optional): The name of the encoding to use. Defaults to "cl100k_base",
            which is the most commonly used encoder (used by GPT-4).

    Returns:
        int: The number of tokens in the input string.

    Example:
        >>> text = "Hello, how are you?"
        >>> num_tokens = num_tokens_in_string(text)
        >>> print(num_tokens)
        5
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens




def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    """
    Calculate the chunk size based on the token count and token limit.

    Args:
        token_count (int): The total number of tokens.
        token_limit (int): The maximum number of tokens allowed per chunk.

    Returns:
        int: The calculated chunk size.

    Description:
        This function calculates the chunk size based on the given token count and token limit.
        If the token count is less than or equal to the token limit, the function returns the token count as the chunk size.
        Otherwise, it calculates the number of chunks needed to accommodate all the tokens within the token limit.
        The chunk size is determined by dividing the token limit by the number of chunks.
        If there are remaining tokens after dividing the token count by the token limit,
        the chunk size is adjusted by adding the remaining tokens divided by the number of chunks.

    Example:
        >>> calculate_chunk_size(1000, 500)
        500
        >>> calculate_chunk_size(1530, 500)
        389
        >>> calculate_chunk_size(2242, 500)
        496
    """

    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size






def translate(
        source_lang,
        target_lang,
        source_text,
        style,
        outline_text,
        country,
        max_tokens=MAX_TOKENS_PER_CHUNK,

):
    """Translate the source_text from source_lang to target_lang."""

    num_tokens_in_text = num_tokens_in_string(source_text)

    if debug:
        ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        if debug:
            ic("Translating text as a single chunk")

        final_translation , outline = one_chunk_translate_text(
            source_lang, target_lang, source_text, style,outline_text, country ,
        )

        return final_translation, outline

    else:
        raise ValueError("Chunks is oversized")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        if debug:
            ic(token_size)

        return
