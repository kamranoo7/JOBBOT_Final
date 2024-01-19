from flask import Flask, render_template, request, redirect, url_for, jsonify, session, Response
import weaviate
import openai
import csv
import gspread
from google.oauth2.service_account import Credentials
import json
import io
import re
import threading

app = Flask(__name__)

key = "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"
lm_client = openai.OpenAI(api_key=key)
credentials_path = "credentials.json"
response = "Done."
app.secret_key = "secret_key"


global intro
intro = """
Welcome to the alpha version of JobBot, created by Capria. JobBot is powered by ChatGPT and can
discuss both opportunities and risks relative to Al automation of your job and implications for your career. Content comes from the writings of authors Ravi Venkatesan and Will Poole as well as select experts. Ask anything about jobs and careers, such as
"What should my daughter study to ensure her job is not replaced by AI?" or "What topics will the authors Ravi's and Will's new book address?" Please use the thumbs-up or down button to give us feedback.
"""

def add_row_to_sheet(data, sheet_id):
    creds = Credentials.from_service_account_file(
        credentials_path, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)

    try:
        sheet = client.open_by_key(sheet_id)
        worksheet = sheet.get_worksheet(0)
        worksheet.append_row(data)
        print("Row added successfully!")
    except Exception as e:
        print("Error: ", e)


layer_1 = weaviate.Client(
    url="https://job1-4aww43nt.weaviate.network",
    # auth_client_secret=weaviate.AuthApiKey(api_key="vvZhMKvXbg2iETqRZ1EyLJ8302jWE436t2oG"),
    additional_headers={
        "X-OpenAI-Api-Key": "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"
    },
)

name_1 = "job1"

layer_2 = weaviate.Client(
    url="https://job2-d1txg4dq.weaviate.network",
    # auth_client_secret=weaviate.AuthApiKey(api_key="vvZhMKvXbg2iETqRZ1EyLJ8302jWE436t2oG"),
    additional_headers={
        "X-OpenAI-Api-Key": "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"
    },
)

name_2 = "job2"

@app.route("/control_panel", methods=["GET", "POST"])
def control_panel():
    global intro
    if request.method == "POST":
        session['language'] = request.form.get('language', 'english')
        session['language'] = get_language(session['language'])

        session['intro'] = request.form.get('intro', '')
        intro = session['intro'] if session['intro'] != '' else intro

        session['prompt_level1'] = request.form.get('prompt_level1', '')
        session['prompt_level2'] = request.form.get('prompt_level2', '')
        session['prompt_level3'] = request.form.get('prompt_level3', '')

    return render_template("control_panel.html", 
                           language=session.get('language', 'english'),
                           intro=intro,
                           prompt_level1=session.get('prompt_level1', ''),
                           prompt_level2=session.get('prompt_level2', ''),
                           prompt_level3=session.get('prompt_level3', ''))

@app.route("/trans")
def trans():
    global intro
    print("/trans-------")
    newList = []
    transwords = ["JobBot", "User", "Enter Your Query", "Feedback", "Fast Mode",intro,"Yes","Submit","Close",'Slow Mode','Groq Mode']

    if session['language'] != "en":
        for item in transwords:
            trans = translate_text(item, session['language'])
            print(trans, "transscripted words ")
            newList.append(trans)

    if not newList:
        newList = transwords

    print(newList, "---------------trans words ")
    return jsonify(newList)

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    try:
        audio_file = request.files['audioFile']
        if audio_file:
                audio_bytes = audio_file.read()
                audio_file = FileWithNames(audio_bytes)
                session["greet"], session["language"] = transcribe(audio_file)
                session['language'] = request.form['language'] if request.form['language'] != 'auto' else session['language']
                session["language"] = get_language(session["language"])
                print("\n\n\n",session["language"],"\n\n\n")

                print("\n\n\n",session["language"],"\n\n\n")
    except Exception as e:
        print(e)
        update_logs(e)
        session["language"] = 'english'

    return jsonify({'channel': 'chat'})

@app.route('/chat')
def chat():
    
    print("Redirecting...")
    return render_template('chat.html')


@app.route("/")
def index():
    # return render_template("orb.html")
    return render_template("orb.html")


custom_functions = [
    {
        "name": "return_response",
        "description": "Function to be used to return the response to the question, and a boolean value indicating if the information given was suffieicnet to generate the entire answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_list": {
                    "type": "array",
                    "description": "List of chunk ids. ONLY the ones used to generate the response to the question being asked. return the id only if the info was used in the response. think carefully.",
                    "items": {"type": "integer"},
                },
                "response": {
                    "type": "string",
                    "description": "This should be the answer that was generated from the context, given the question",
                },
                "sufficient": {
                    "type": "boolean",
                    "description": "This should represent wether the information present in the context was sufficent to answer the question. Return True is it was, else False.",
                },
            },
            "required": ["response", "sufficient", "item_list"],
        },
    }
]

custom_functions_1 = [
    {
        "name": "return_response",
        "description": "Function to be used to return the response to the question, and a boolean value indicating if the information given was suffieicnet to generate the entire answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "boolean",
                    "description": "This should be the answer that was generated from the context, given the question",
                },
                "sufficient": {
                    "type": "boolean",
                    "description": "This should represent wether the information present in the context was sufficent to answer the question. Return True is it was, else False.",
                },
            },
            "required": ["response", "sufficient"],
        },
    }
]

import time


def ask_gpt(question, context, gpt, addition):
    user_message = "Question: \n\n" + question + "\n\n\nContext: \n\n" + context
    system_message = "You will be given context from several pdfs, this context is from several chunks, rettrived from a vector DB. each chunk will have a chunk id above it. You will also be given a question. Formulate an answer, ONLY using the context, and nothing else. provide in-text citations within square brackets at the end of each sentence, right after each fullstop. The citation number represents the chunk id that was used to generate that sentence. Do Not bunch multiple citations in one bracket. Uee seperate brackets for each digit. {} Return the response along with a boolean value indicating if the information from the context was enough to answer the question. Return true if it was, False if it wasnt. Return the response, which is th answer to the question asked".format(addition)

    msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    def call_api():
        nonlocal args
        response = lm_client.chat.completions.create(
            model=gpt,
            messages=msg,
            max_tokens=4000,
            temperature=0.0,
            functions=custom_functions,
            function_call="auto",
        )
        args = response

    args = None
    thread = threading.Thread(target=call_api)
    start = time.time()
    thread.start()
    thread.join(timeout=160)

    if thread.is_alive():
        print(time.time() - start)
        return (
            "Timeout. Pranav has set level timeout to 160  seconds. The timeout error usually happens when the API has been used multiple times.",
            False,
            [0, 1],
        )

    response = args
    reply = response.choices[0].message.content
    item_list = []
    sufficient = False
    print("This is as close as possible")
    if reply is None:
        reply = json.loads(response.choices[0].message.function_call.arguments)[
            "response"
        ]
        print("This is as close as possible")

        sufficient = json.loads(response.choices[0].message.function_call.arguments)[
            "sufficient"
        ]
        print("This is as close as possible")

    return reply, sufficient


def ask_gpt_fast(question, context, addition):
    user_message = "Question: \n\n" + question + "\n\n\nContext: \n\n" + context
    system_message = "You will be given context from several pdfs, this context is from several chunks, rettrived from a vector DB. each chunk will have a chunk id above it. You will also be given a question. Formulate an answer, ONLY using the context, and nothing else. {} Return the text response and a boolean value indicating if the information from the context was enough to answer the question. Return true if it was, False if it wasnt. Return the response, which is the answer to the question asked. If the answer cannot be formulated using the context, say that it is not possile. The reader of your response does not have any idea of the 'context' being passed. Do not reference the presence of the context in the final response, just provide the answer directly.".format(addition)

    msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    response = lm_client.chat.completions.create(
        model="gpt-4",
        messages=msg,
        max_tokens=2000,
        temperature=0.0,
        functions=custom_functions_1,
        function_call="auto",
    )

    reply = response.choices[0].message.content
    sufficient = False
    if reply is None:
        reply = json.loads(response.choices[0].message.function_call.arguments)[
            "response"
        ]
        sufficient = json.loads(response.choices[0].message.function_call.arguments)[
            "sufficient"
        ]
    return reply, sufficient


def ask_gpt_generic(question):
    user_message = question
    system_message = (
        "Answer the question."+ session["generic_system_msg_level_3"]
    )
    msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    response = lm_client.chat.completions.create(
        model="gpt-3.5-turbo-16k", messages=msg, max_tokens=4000, temperature=0.0
    )
    reply = response.choices[0].message.content
    return reply



def qdb(query, db_client, name, cname):
    context = None
    metadata = []
    try:
        limit = 5
        res = (
            db_client.query.get(name, ["text", "metadata"])
            .with_near_text({"concepts": query})
            .with_limit(limit)
            .do()
        )
        context = ""
        metadata = []
        chunk_id = 0
        for i in range(limit):
            context += "Chunk ID: " + str(chunk_id) + "\n"
            context += res["data"]["Get"][cname][i]["text"] + "\n\n"
            metadata.append(res["data"]["Get"][cname][i]["metadata"])
            chunk_id += 1
    except Exception as e:
        print("Exception in DB, dude.")
        print(e)
        time.sleep(3)
        context, metadata = qdb(query, db_client, name, cname)
    return context, metadata


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    print(data)
    try:
        unique_id = data.get('uniqueId')
        thumbs = data.get('type', 'Text')
        l2 = data.get('l2ResponseClicked')
        l3 = data.get('l3ResponseClicked')
        feedback_text = data.get('feedback', 'Null')
        level = data.get('level', 'test')
        print(thumbs, feedback_text, level)
        if not l2 and not l3:
            add_row_to_sheet([session['transcription'], session['level_1_response'], thumbs,feedback_text], "1OvOj468hgwhjrBFqWrHrZtSirrodrUEejBUUr37by_Y")
        if l2 and not l3:
            add_row_to_sheet([session['transcription'], session['level_2_response'], thumbs,feedback_text], "1OvOj468hgwhjrBFqWrHrZtSirrodrUEejBUUr37by_Y")
        if l2 and l3:
            add_row_to_sheet([session['transcription'], session['level_3_response'], thumbs,feedback_text], "1OvOj468hgwhjrBFqWrHrZtSirrodrUEejBUUr37by_Y")
    except Exception as e:
        update_logs(e)

    return jsonify({"status": "success"})


def transcribe(audio_file):
    try:
        response = lm_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
        print(response)
        transcription = response.text
        language = response.text + " " + response.language
        # print(language)
    except Exception as e:
        print(e)
        update_logs(e)
        transcription = "Error."
        language = 'english'

    return transcription, language


class FileWithNames(io.BytesIO):
    name = "audio.wav"

import os
from datetime import datetime

def update_logs(input_string):
    file_exists = os.path.isfile('logs.txt')

    with open('logs.txt', 'a' if file_exists else 'w') as file:
        if file_exists:
            file.write('\n\n\n\n')
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file.write(f'{current_time}\n{input_string}\n')


def process_response(input_string, replacements):
    def replacement(match):
        index = int(match.group(1))
        return (
            f"[{replacements[index]}]" if index < len(replacements) else match.group(0)
        )

    try:
        return re.sub(r"\[(\d+)\]", replacement, input_string)
    except:
        return input_string

import requests

def translate_text(text, target_language):
    print(target_language)
    api_key = 'AIzaSyAtfrkxLhTygIJi9Rb-l0duA8fV9LgKZ7M'  # Replace with your API key

    url = 'https://translation.googleapis.com/language/translate/v2'
    data = {
        'q': text,
        'target': target_language,
        'format': 'text'
    }
    headers = {
        'Content-Type': 'application/json'
    }
    params = {
        'key': api_key
    }
    response = requests.post(url, headers=headers, params=params, json=data)
    r =  response.json()
    print(r)
    return r['data']['translations'][0]['translatedText']

def get_language(lang):
    print("getting lang.")
    lang = lang.lower()
    if 'arabic' in lang: return 'ar'
    if 'kannada' in lang: return 'kn'
    if 'telugu' in lang: return 'te'
    if 'spanish' in lang: return 'es'
    if 'hebrew' in lang: return 'he'
    if 'japanese' in lang: return 'ja'
    if 'korean' in lang: return 'ko'
    if 'hindi' in lang: return 'hi'
    if 'bengali' in lang: return 'bn'
    if 'tamil' in lang: return 'ta'
    if 'urdu' in lang: return 'ur'
    if 'chinese' in lang: return 'zh-CN'
    if 'french' in lang: return 'fr'
    if 'german' in lang: return 'de'
    
    session['language'] = 'english'
    return 'en'

import requests
import json


def groq(question, context, start, ending, language, first=False):
    api_key = 'aeb9ooc4eiTiedootee3dei4aipuub9v'

    url = 'https://api.groq.com/v1/request_manager/text_completion'

    if context:
        system_message = 'You are an assitant, that answers the question, only from the context provided. Only answer questions that can be answered using the context. Do not let the user know you are referencing context. Remember, the answer that you generate must be grounded in the context provided. Provide partial answers too. Remeber, the most important point is that you must not let the user know that you are referencing a context.It must look asthough the answer is your own work. Which means you CANNNOT say things like "based on the provided context", or anything similar.'
        # system_message = 'You are an assitant, that answers the question, only from the context provided. If the answer cannot be formulated from the given context, you will say so. remember, the answer that you generate must be grounded in the context provided. Provide partial answers too. Remeber, the most important point is that you must not let the user know that you are referencing a context.It must look asthough the answer is your own work. Which means you CANNNOT say things like "based on the provided context", or anything similar.'
        user_message = "question: " + question + "\n\n context: \n" + context
    else:
        system_message = 'you are a helpful assistant.'
        user_message = question

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    data = {
        'model_id': 'llama2-70b-4096',
        'system_prompt': system_message,
        'user_prompt': user_message,
        'temperature':0.1,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    response_lines = response.text.strip().split('\n')

    parsed_responses = [json.loads(line) for line in response_lines]

    concatenated_content = ""

    for parsed_response in parsed_responses:
        content = parsed_response.get('result', {}).get('content', '')
        concatenated_content += content

    if first:
        concatenated_content = start + concatenated_content
    else:
        pass

    concatenated_content += ending

    if language != 'en':
        try: 
            concatenated_content = translate_text(concatenated_content, language)
        except Exception as e:
            print(e)
            pass
    if not first: concatenated_content = start + concatenated_content
    data = {"response": concatenated_content, "sufficient": False, "endOfStream": True}
    json_data = json.dumps(data)
    yield f"data: {json_data}\n\n"

@app.route('/level1', methods=['POST'])
def level1():
    print('level 1....\n\n\n')
    session["transcription"] = request.form["query"] if "query" in request.form else ""
    session["generic_system_msg_level_1"] = (
        request.form["lmprompt1"] if "lmprompt1" in request.form else ""
    )
    if request.form["leng"] != "": session["language"] = request.form["leng"]

    session["language"] = (
        "english" if session["language"] == "" else session["language"]
    )

    if request.form['fast'] == 'true':
        session['toggle'] = 'fast'
    if request.form['slow'] == 'true':
        session['toggle'] = 'slow'
    if request.form['groq'] == 'true':
        session['toggle'] = 'groq'

    audio_file = request.files["audio"] if "audio" in request.files else None

    try:
        if audio_file:
            print(audio_file, request.files["audio"])
            audio_bytes = audio_file.read()
            print("-----------------------------------------------------------------------------")
            audio_file = FileWithNames(audio_bytes)
            print("-----------------------------------------------------------------------------")
            session["transcription"], session['language'] = transcribe(audio_file)
            session['language'] = get_language(session['language'])

            print("\n\n\n",session['language'],"\n\n\n")

            if session['language'].lower() != 'en':
                session["transcription"] = translate_text(session["transcription"], 'en')
                print("\n\n\n\nEnglish:  ",session["transcription"], session['language'])
    except Exception as e:
        session['language'] = 'en'
        print(e)
        update_logs(e)
        session["transcription"] = "Error."
    return jsonify({"message": "Data received, start streaming"})


def generate_level1(question, context, model, addition, language, start, ending, first=False):
        if context:
            user_message = "Question: \n\n" + question + "\n\n\nContext: \n\n" + context
            system_message = "You will be given context from several pdfs, this context is from several chunks, rettrived from a vector DB. each chunk will have a chunk id above it. You will also be given a question. Formulate an answer, ONLY using the context, and nothing else. {}. Return the response, which is the answer to the question asked. If the answer cannot be formulated using the context, say that it is not possile. The reader of your response does not have any idea of the 'context' being passed. Do not reference the presence of the context in the final response, just provide the answer directly.".format(addition)
        else:
            user_message = question
            system_message = "Answer the question."+ addition
        
        translate = ""
        response = start if first else ""

        
        msg = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        
        stream = lm_client.chat.completions.create(
            model=model,
            messages=msg,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
                translate = response
                if language != "en":
                    try: 
                        translate = translate_text(response, language)
                    except Exception as e:
                        print(e)
                        pass
                else:
                    pass
                if not first: translate = start + translate
                data = {"response": translate, "sufficient": False}
                json_data = json.dumps(data)
                yield f"data: {json_data}\n\n"
        
        ending = translate_text(ending, language) if language != 'en' else ending
        translate += ending
        data = {"response": translate, "sufficient": False, "endOfStream": True}
        json_data = json.dumps(data)
        yield f"data: {json_data}\n\n"


@app.route('/level1/stream')
def level1_stream():
    try:
        context, metadata = qdb(session["transcription"], layer_1, "job1", "Job1")
        sufficient = False
    except Exception as e:
        update_logs(e)
        context = "No context"
        metadata = ["1"]

    response = "This is what the authors Ravi and Will have to say:\n\n" 
    ending = "\n\n Would you like to see what other experts have to say about it?\n\n"

    try:
        if session['toggle'] == "slow":
            resp  = Response(generate_level1(session["transcription"],context,"gpt-4",session["generic_system_msg_level_1"], session['language'], response, ending, True), content_type='text/event-stream')
            return resp
        
        elif session['toggle'] == "fast":
            resp = Response(generate_level1(session["transcription"],context,"gpt-3.5-turbo-16k",session["generic_system_msg_level_1"], session['language'], response, ending, True), content_type='text/event-stream')
            return resp
        
        else:
            resp =  Response(groq(session["transcription"],context, response, ending, session['language']), content_type='text/event-stream')
            return resp
    
    except:
        data = {"response": "Error.", "sufficient": False}
        json_data = json.dumps(data)
        resp = "data: {json_data}\n\n"
        return Response(resp, content_type='text/event-stream')


@app.route('/level2', methods=['POST'])
def level2():
    session["generic_system_msg_level_2"] = (
            request.form["lmprompt2"] if "lmprompt2" in request.form else ""
        )
    session['layer_1_response'] = request.form['response']
    return jsonify({"message": "Data received, start streaming"})
    
@app.route('/level2/stream')
def level2_stream():
    try:
        context, metadata = qdb(session["transcription"], layer_2, "job2", "Job2")
        sufficient = False
    except Exception as e:
        update_logs(e)
        context = "No context"
        metadata = ["1"]

    session['response'] = session['layer_1_response']
    ending =  "\n\nWould you like to see what ChatGPT 3.5 has to say about this?\n\n"

    try:
        if session['toggle'] == "slow":
            resp  = Response(generate_level1(session["transcription"],context,"gpt-4",session["generic_system_msg_level_2"], session['language'], session['response'], ending), content_type='text/event-stream')
            return resp
        
        elif session['toggle'] == "fast":
            resp = Response(generate_level1(session["transcription"],context,"gpt-3.5-turbo-16k",session["generic_system_msg_level_2"], session['language'], session['response'], ending), content_type='text/event-stream')
            return resp
        
        else:
            ending =  "\n\nWould you like to see what Llama-2 powered by Groq has to say about this?\n\n"
            resp =  Response(groq(session["transcription"],context, session['response'], ending, session['language']), content_type='text/event-stream')
            return resp
    
    except:
        data = {"response": "Error.", "sufficient": False}
        json_data = json.dumps(data)
        resp = "data: {json_data}\n\n"
        return Response(resp, content_type='text/event-stream')


@app.route('/level3', methods=['POST'])
def level3():
    session["generic_system_msg_level_3"] = (
            request.form["lmprompt3"] if "lmprompt3" in request.form else ""
        )
    
    print(request.form)
    session['layer_2_response'] = request.form['response']
    session['response'] = session['layer_2_response']

    return jsonify({"message": "Data received, start streaming"})
    

@app.route('/level3/stream')
def level3_stream():
    try:
        if session['toggle'] == "groq":
            resp =  Response(groq(session["transcription"],None, session['response'], "", session['language']), content_type='text/event-stream')
            return resp
        else:
            resp  = Response(generate_level1(session["transcription"],None,"gpt-3.5-turbo-16k",session["generic_system_msg_level_3"], session['language'],  session['response'], ""), content_type='text/event-stream')
            return resp
    except:
        data = {"response": "Error.", "sufficient": False}
        json_data = json.dumps(data)
        resp = "data: {json_data}\n\n"
        return Response(resp, content_type='text/event-stream')


# @app.route('/level2', methods=['POST'])
# def level2():
#     def generate_level1():
#         # Assume get_level1_data() is a function that yields parts of the response
#         for part in range(10):
#             yield f"data: {part}\n\n"  # Format for Server-Sent Events

#     return Response(generate_level1(), content_type='text/event-stream')

# @app.route('/level3', methods=['POST'])
# def level3():
#     def generate_level1():
#         # Assume get_level1_data() is a function that yields parts of the response
#         for part in range(10):
#             yield f"data: {part}\n\n"  # Format for Server-Sent Events

#     return Response(generate_level1(), content_type='text/event-stream')



# @app.route("/level1", methods=["POST"])
# def level1():
#     print('level 1....\n\n\n')
#     session["transcription"] = request.form["query"] if "query" in request.form else ""
#     session["generic_system_msg_level_1"] = (
#         request.form["lmprompt1"] if "lmprompt1" in request.form else ""
#     )
#     if request.form["leng"] != "": session["language"] = request.form["leng"]

#     session["language"] = (
#         "english" if session["language"] == "" else session["language"]
#     )

#     if request.form['fast'] == 'true':
#         session['toggle'] = 'fast'
#     if request.form['slow'] == 'true':
#         session['toggle'] = 'slow'
#     if request.form['groq'] == 'true':
#         session['toggle'] = 'groq'

#     audio_file = request.files["audio"] if "audio" in request.files else None

#     try:
#         if audio_file:
#             print(audio_file, request.files["audio"])
#             audio_bytes = audio_file.read()
#             print("-----------------------------------------------------------------------------")
#             audio_file = FileWithNames(audio_bytes)
#             print("-----------------------------------------------------------------------------")
#             session["transcription"], session['language'] = transcribe(audio_file)
#             session['language'] = get_language(session['language'])

#             print("\n\n\n",session['language'],"\n\n\n")

#             if session['language'].lower() != 'en':
#                 session["transcription"] = translate_text(session["transcription"], 'en')
#                 print("\n\n\n\nEnglish:  ",session["transcription"], session['language'])
#     except Exception as e:
#         session['language'] = 'en'
#         print(e)
#         update_logs(e)
#         session["transcription"] = "Error."
    
#     try:
#         context, metadata = qdb(session["transcription"], layer_1, "job1", "Job1")
#         sufficient = False
#     except Exception as e:
#         update_logs(e)
#         context = "No context"
#         metadata = ["1"]

#     try:
#         if session['toggle'] == "slow":
#             response, sufficient = ask_gpt(
#                 session["transcription"],
#                 context,
#                 "gpt-4",
#                 session["generic_system_msg_level_1"],
#             )
#         elif session['toggle'] == "fast":
#             response, sufficient = ask_gpt_fast(
#                 session["transcription"],
#                 context,
#                 session["generic_system_msg_level_1"],
#             )
#         else:
#             response = groq(session["transcription"], context)
#             sufficient = True

#         sufficient = True
#         if sufficient:
#             response = process_response(response, metadata)
#             print("\n\n\n\nDone\n\n\n\n")
#             response = "This is what the authors Ravi and Will have to say:\n\n" + response
#         else:
#             print(response)
#             response = "Authors Ravi and Will do not have anything to say about what you asked, yet."    

#         response += "\n\n Would you like to see what other experts have to say about it?"

#         if session['language'] != "en":

#             try: 
#                 response = translate_text(response, session['language'])
#             except:
#                 pass
#     except Exception as e:
#         update_logs(e)
#         response = "There was an error."
        
#     session['level_1_response'] = response

#     return jsonify({"response": response, "sufficient": False})


# @app.route("/level2", methods=["POST"])
# def level2():

#     print("Level2......\n")
#     sufficient = False
#     session["generic_system_msg_level_2"] = (
#         request.form["lmprompt2"] if "lmprompt2" in request.form else ""
#     )
#     try:
#         context, metadata = qdb(session["transcription"], layer_2, "job2", "Job2")
#         if session['toggle'] == "slow":
#             print("This is slow.")
#             response, sufficient = ask_gpt(
#                 session["transcription"],
#                 context,
#                 "gpt-4",
#                 session["generic_system_msg_level_2"],
#             )
#             print('Slow is done.')
#         elif session['toggle'] == "fast":
#             response, sufficient = ask_gpt_fast(
#                 session["transcription"],
#                 context,
#                 session["generic_system_msg_level_2"],
#             )
#         else:
#             response = groq(session["transcription"], context)
#             sufficient = True

#         sufficient = True

#         if sufficient:
#             response = process_response(response, metadata)
#             response = "Here's what other experts have to say on the matter::\n\n" + response
#         else:
#             response = "Other experts have no comments on this topic. Refer to the standard Language Model response below:"

#         lm = "LLama2-Groq" if session['toggle']=='groq' else 'GPT-3.5'
#         response += "\n\nWould you like to see what {} thinks about what you asked?".format(lm)
#     except Exception as e:
#         update_logs(e)
#         print(e)
#         response = "There was an error."
    
#     session['level_2_response'] = response
    
#     if session['language'] != "en":
#         try: 
#             response = translate_text(response, session['language'])
#         except Exception as e:
#             update_logs(e)
#             response = "There was an error."
#     # time.sleep(6)
#     print("Done.")

#     return jsonify({"response": response, "sufficient": False})


# @app.route("/level3", methods=["POST"])
# def level3():
#     print("Level  3......\n")
#     session["generic_system_msg_level_3"] = (
#         request.form["lmprompt3"] if "lmprompt2" in request.form else ""
#     )
#     try:
#         if session['toggle'] == "groq":
#             response = groq(session["transcription"], None)
#         else:
#             response = ask_gpt_generic(session["transcription"])
#         # response = "Generic Language Model response:\n\n" + response
#         response = "\n\n" + response
#     except Exception as e:
#         update_logs(e)
#         response = "There was an error."

#     if session['language'] != "en":
#         try: 
#             response = translate_text(response, session['language'])
#         except Exception as e:
#             update_logs(e)
#             response = "There was an error."

#     session['level_3_response'] = response

#     return jsonify({"response": response, "sufficient": True})



if __name__ == "__main__":
    app.run(debug=True)
