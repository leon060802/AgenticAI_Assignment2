import platform
import argparse
import time
import json
import re
import os
import shutil
import logging
import google.generativeai as genai
from PIL import Image
import io
import base64

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_TEXT_ONLY, SYSTEM_PREVIOUS_STEP, ERROR_GROUNDING_AGENT_PROMPT
from utils import get_web_element_rect, encode_image, extract_information, print_message,\
    get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, clip_message_and_obs, clip_message_and_obs_text_only

def setup_logger(folder_path):
    log_file_path = os.path.join(folder_path, 'agent.log')

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_file_path, encoding="utf-8")
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def driver_config(args):
    options = webdriver.ChromeOptions()
    options.add_argument("disable-blink-features=AutomationControlled")

    if args.save_accessibility_tree:
        args.force_device_scale = True

    if args.force_device_scale:
        options.add_argument("--force-device-scale-factor=1")
    if args.headless:
        options.add_argument("--headless")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    options.add_experimental_option(
        "prefs", {
            "download.default_directory": args.download_dir,
            "plugins.always_open_pdf_externally": True
        }
    )
    return options


def format_msg(it, init_msg, pdf_obs, warn_obs, web_img_b64, web_text, prev_step_action=""):
    if it == 1:
        init_msg += f"{prev_step_action}\nI've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
        # Format for Gemini API
        return [
            {"text": init_msg},
            {"inline_data": {"mime_type": "image/png", "data": web_img_b64}}
        ]
    else:
        if not pdf_obs:
            message_text = f"{prev_step_action}\nObservation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
        else:
            message_text = f"{prev_step_action}\nObservation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
        
        # Format for Gemini API
        return [
            {"text": message_text},
            {"inline_data": {"mime_type": "image/png", "data": web_img_b64}}
        ]


def format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree, prev_step_action=""):
    if it == 1:
        # Format for Gemini API
        return [{"text": init_msg + '\n' + ac_tree}]
    else:
        if not pdf_obs:
            message_text = f"{prev_step_action}\nObservation:{warn_obs} please analyze the accessibility tree and give the Thought and Action.\n{ac_tree}"
        else:
            message_text = f"{prev_step_action}\nObservation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree of the current page is also given, give the Thought and Action.\n{ac_tree}"
        
        # Format for Gemini API
        return [{"text": message_text}]


def call_gemini_api(args, model, messages):
    retry_times = 0
    while True:
        try:
            if not args.text_only:
                logging.info('Calling Gemini API...')
                # For Gemini API, we need to flatten the messages into a single list of parts
                flattened_messages = []
                for msg in messages:
                    if isinstance(msg, list):
                        flattened_messages.extend(msg)
                    else:
                        flattened_messages.append(msg)
                
                response = model.generate_content(flattened_messages)
            else:
                logging.info('Calling Gemini API (text only)...')
                # For text-only mode, flatten the messages
                flattened_messages = []
                for msg in messages:
                    if isinstance(msg, list):
                        flattened_messages.extend(msg)
                    else:
                        flattened_messages.append(msg)
                
                response = model.generate_content(flattened_messages)

            gemini_call_error = False
            return response

        except Exception as e:
            logging.info(f'Error occurred, retrying. Error type: {type(e).__name__}')
            retry_times += 1
            if retry_times > 3:
                logging.error('Max retries reached. Exiting...')
                raise e
            time.sleep(5)


def exec_action_click(info, web_ele, driver_task):
    driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
    web_ele.click()
    time.sleep(3)


def exec_action_type(info, web_ele, driver_task):
    warn_obs = ""
    type_content = info['content']

    ele_tag_name = web_ele.tag_name.lower()
    ele_type = web_ele.get_attribute("type")
    # outer_html = web_ele.get_attribute("outerHTML")
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
    try:
        # Not always work to delete
        web_ele.clear()
        # Another way to delete
        if platform.system() == 'Darwin':
            web_ele.send_keys(Keys.COMMAND + "a")
        else:
            web_ele.send_keys(Keys.CONTROL + "a")
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except:
        pass

    actions = ActionChains(driver_task)
    actions.click(web_ele).perform()
    actions.pause(1)

    try:
        driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
    except:
        pass

    actions.send_keys(type_content)
    actions.pause(2)

    actions.send_keys(Keys.ENTER)
    actions.perform()
    time.sleep(10)
    return warn_obs


def exec_action_scroll(info, web_eles, driver_task, args, obs_info):
    scroll_ele_number = info['number']
    scroll_content = info['content']
    if scroll_ele_number == "WINDOW":
        if scroll_content == 'down':
            driver_task.execute_script(f"window.scrollBy(0, {args.window_height*2//3});")
        else:
            driver_task.execute_script(f"window.scrollBy(0, {-args.window_height*2//3});")
    else:
        if not args.text_only:
            scroll_ele_number = int(scroll_ele_number)
            web_ele = web_eles[scroll_ele_number]
        else:
            element_box = obs_info[scroll_ele_number]['union_bound']
            element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
            web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
        actions = ActionChains(driver_task)
        driver_task.execute_script("arguments[0].focus();", web_ele)
        if scroll_content == 'down':
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
        else:
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
    time.sleep(3)


def setup_gemini_model(api_key):
    genai.configure(api_key=api_key)
    # Set up the model with specific parameters for gemini-1.5-flash
    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",  # Updated to use flash model
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    return model


def process_image_for_gemini(image_data):
    try:
        # Remove the data URL prefix if present
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Open as PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return image
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/tasks_test.jsonl')
    parser.add_argument('--max_iter', type=int, default=15)
    parser.add_argument('--trajectory', action='store_true')
    parser.add_argument('--error_max_reflection_iter', type=int, default=1, help='Number of reflection restarts allowed when exceeding max_iter')
    
    parser.add_argument("--api_key", type=str, required=True, help="API key for Gemini")
    parser.add_argument("--api_model", type=str, default="gemini-1.5-flash", help="Gemini model to use")
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_attached_imgs", type=int, default=10, help="Maximum number of attached images")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--download_dir", type=str, default="downloads")
    parser.add_argument("--text_only", action='store_true', help="Use text-only mode")
    # for web browser
    parser.add_argument("--headless", action='store_true', help='Run in headless mode')
    parser.add_argument("--save_accessibility_tree", action='store_true', help="Save accessibility tree")
    parser.add_argument("--force_device_scale", action='store_true', help="Force device scale factor")
    parser.add_argument("--window_width", type=int, default=1024)
    parser.add_argument("--window_height", type=int, default=768)  # for headless mode, there is no address bar
    parser.add_argument("--fix_box_color", action='store_true')
    parser.add_argument("--start_maximized", action='store_true')

    args = parser.parse_args()

    # Setup logging
    setup_logger(args.output_dir)

    # Setup Gemini model
    model = setup_gemini_model(args.api_key)

    # Setup Chrome driver
    options = driver_config(args)
    driver_task = webdriver.Chrome(options=options)

    # Save Result file
    current_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    result_dir = os.path.join(args.output_dir, current_time)
    os.makedirs(result_dir, exist_ok=True)

    # Load tasks
    tasks = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))


    for task_id in range(len(tasks)):
        task = tasks[task_id]
        task_dir = os.path.join(result_dir, 'task{}'.format(task["id"]))
        os.makedirs(task_dir, exist_ok=True)
        setup_logger(task_dir)
        logging.info(f'########## TASK{task["id"]} ##########')

        # Initialize the window (The window was too small, and I couldn't see some parts of the screen, so I added the --start_maximized parameter).
        driver_task.maximize_window()
        driver_task.set_window_size(args.window_width, args.window_height)  # larger height may contain more web information
        

        # About window size, 765 tokens
        # You can resize to height = 512 by yourself (255 tokens, Maybe bad performance)
        driver_task.get(task['web'])
        try:
            driver_task.find_element(By.TAG_NAME, 'body').click()
        except:
            pass
        # sometimes enter SPACE, the page will sroll down
        driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
        time.sleep(5)

        # We only deal with PDF file
        for filename in os.listdir(args.download_dir):
            file_path = os.path.join(args.download_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        download_files = []  # sorted(os.listdir(args.download_dir))

        fail_obs = ""  # When error execute the action
        pdf_obs = ""  # When download PDF file
        warn_obs = ""  # Type warning
        pattern = r'Thought:|Action:|Observation:|Errors:|Explanation:'

        # Initialize messages with system prompt
        messages = []
        obs_prompt = "Observation: please analyze the attached screenshot and give the Thought and Action. "
        if args.text_only:
            obs_prompt = "Observation: please analyze the accessibility tree and give the Thought and Action."

        init_msg = f"""Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. \n"""
        init_msg = init_msg.replace('https://www.example.com', task['web'])
        init_msg = init_msg + obs_prompt

        # Add system prompt to messages
        if args.text_only:
            messages.append([{"text": SYSTEM_PROMPT_TEXT_ONLY}])
        else:
            messages.append([{"text": SYSTEM_PROMPT}])

        it = 0
        accumulate_prompt_token = 0
        accumulate_completion_token = 0

        # Error Grounding Agent
        activate_EGA=True
        error_exist=False
        EGA_explanation=""
        bot_thought=""
        
        # Reflection: Trajectory
        current_history = ""     # Record the steps of the current iteration.
        
        print(f"Trajectory: {args.trajectory}")
        print(f"EGA: {activate_EGA}")
        
        while it < args.max_iter:
            logging.info(f'Iter: {it}')
            it += 1
            if not fail_obs:
                try:
                    if not args.text_only:
                        rects, web_eles, web_eles_text = get_web_element_rect(driver_task, fix_color=args.fix_box_color)
                    else:
                        accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                        ac_tree, obs_info = get_webarena_accessibility_tree(driver_task, accessibility_tree_path)

                except Exception as e:
                    if not args.text_only:
                        logging.error('Driver error when adding set-of-mark.')
                    else:
                        logging.error('Driver error when obtaining accessibility tree.')
                    logging.error(e)
                    break
            
                img_path = os.path.join(task_dir, 'screenshot{}.png'.format(it))
                driver_task.save_screenshot(img_path)
                
                # encode image
                b64_img = encode_image(img_path)

                # Error Grounding Agent
                if it>1 and activate_EGA:
                    # Format for Gemini API
                    EGA_messages = [
                        {"text": ERROR_GROUNDING_AGENT_PROMPT},
                        {"text": "Thought:"+bot_thought+"\nScreenshot:"},
                        {"inline_data": {"mime_type": "image/png", "data": b64_img}}
                    ]
                    
                    response = call_gemini_api(args, model, EGA_messages)
                    if response:
                        accumulate_prompt_token += len(str(response))
                        accumulate_completion_token += len(str(response))
                        logging.info(f'Accumulate Prompt Tokens: {accumulate_prompt_token}; Accumulate Completion Tokens: {accumulate_completion_token}')
                        logging.info('API call complete...')
                    gpt_4v_res = response.text
                    if re.split(pattern, gpt_4v_res)[1].strip() == 'Yes':
                        error_exist = True
                    elif re.split(pattern, gpt_4v_res)[1].strip() == 'No':
                        error_exist = False
                    else:
                        error_exist = False
                        print("error_exist got unexpected result:",gpt_4v_res)
                    if error_exist==True:
                        EGA_explanation = re.split(pattern, gpt_4v_res)[2].strip()
                        
                # accessibility tree
                if (not args.text_only) and args.save_accessibility_tree:
                    accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                    get_webarena_accessibility_tree(driver_task, accessibility_tree_path)

                # format msg
                if not args.text_only:
                    curr_msg = format_msg(it, init_msg, pdf_obs, warn_obs, b64_img, web_eles_text, SYSTEM_PREVIOUS_STEP + current_history)
                    if error_exist == True:
                        curr_msg[0]['text']+=("\nAdditional Information: Looks like your previous thought has some problem in operation. Here is the message from Error Grounding Agent\n"+EGA_explanation)
                else:
                    curr_msg = format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree, SYSTEM_PREVIOUS_STEP + current_history)
                    if error_exist == True:
                        curr_msg[0]['text']+=("\nAdditional Information: Looks like your previous thought has some problem in operation. Here is the message from Error Grounding Agent\n"+EGA_explanation)
                messages.append(curr_msg)
            else:
                curr_msg = [{"text": fail_obs}]
                messages.append(curr_msg)

            # Clip messages, too many attached images may cause confusion
            if not args.text_only:
                messages = clip_message_and_obs(messages, args.max_attached_imgs)
            else:
                messages = clip_message_and_obs_text_only(messages, args.max_attached_imgs)

            # Call Gemini API
            response = call_gemini_api(args, model, messages)
        
            if response:
                accumulate_prompt_token += len(str(response))
                accumulate_completion_token += len(str(response))
                logging.info(f'Accumulate Prompt Tokens: {accumulate_prompt_token}; Accumulate Completion Tokens: {accumulate_completion_token}')
                logging.info('API call complete...')
            gpt_4v_res = response.text
            # Append the response in the correct format for Gemini API
            messages.append([{"text": gpt_4v_res}])
            # print(gpt_4v_res)

            # remove the rects on the website
            if (not args.text_only) and rects:
                logging.info(f"Num of interactive elements: {len(rects)}")
                for rect_ele in rects:
                    driver_task.execute_script("arguments[0].remove()", rect_ele)
                rects = []
                # driver_task.save_screenshot(os.path.join(task_dir, 'screenshot{}_no_box.png'.format(it)))


            # extract action info
            try:
                assert 'Thought:' in gpt_4v_res and 'Action:' in gpt_4v_res
            except AssertionError as e:
                logging.error(e)
                fail_obs = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
                continue
            
            # print(f"Gemini Response: {gpt_4v_res}\n--------")

            bot_thought = re.split(pattern, gpt_4v_res)[1].strip()
            chosen_action = re.split(pattern, gpt_4v_res)[2].strip()
            
            trajectory_info = f"Thought {bot_thought}\nAction {chosen_action}"
            error_info = f"Error: {error_exist}\nExplanation: {EGA_explanation}"
                
            if args.trajectory:
                current_history += trajectory_info
                if activate_EGA:
                    current_history += error_info
                
            print(f"Step {it}:\n{error_info}\n{trajectory_info}\n----")
                
            # logging.info(gpt_4v_res)
            action_key, info = extract_information(chosen_action)

            fail_obs = ""
            pdf_obs = ""
            warn_obs = ""
            # execute action
            try:
                window_handle_task = driver_task.current_window_handle
                driver_task.switch_to.window(window_handle_task)

                if action_key == 'click':
                    if not args.text_only:
                        click_ele_number = int(info[0])
                        web_ele = web_eles[click_ele_number]
                    else:
                        click_ele_number = info[0]
                        element_box = obs_info[click_ele_number]['union_bound']
                        element_box_center = (element_box[0] + element_box[2] // 2,
                                            element_box[1] + element_box[3] // 2)
                        web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])

                    ele_tag_name = web_ele.tag_name.lower()
                    ele_type = web_ele.get_attribute("type")

                    exec_action_click(info, web_ele, driver_task)

                    # deal with PDF file
                    current_files = sorted(os.listdir(args.download_dir))
                    if current_files != download_files:
                        # wait for download finish
                        time.sleep(10)
                        current_files = sorted(os.listdir(args.download_dir))

                        current_download_file = [pdf_file for pdf_file in current_files if pdf_file not in download_files and pdf_file.endswith('.pdf')]
                        if current_download_file:
                            pdf_file = current_download_file[0]
                            pdf_obs = get_pdf_retrieval_ans_from_assistant(model, os.path.join(args.download_dir, pdf_file), task['ques'])
                            shutil.copy(os.path.join(args.download_dir, pdf_file), task_dir)
                            pdf_obs = "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: " + pdf_obs
                        download_files = current_files

                    if ele_tag_name == 'button' and ele_type == 'submit':
                        time.sleep(10)

                elif action_key == 'wait':
                    time.sleep(5)

                elif action_key == 'type':
                    if not args.text_only:
                        type_ele_number = int(info['number'])
                        web_ele = web_eles[type_ele_number]
                    else:
                        type_ele_number = info['number']
                        element_box = obs_info[type_ele_number]['union_bound']
                        element_box_center = (element_box[0] + element_box[2] // 2,
                                            element_box[1] + element_box[3] // 2)
                        web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])

                    warn_obs = exec_action_type(info, web_ele, driver_task)
                    if 'wolfram' in task['web']:
                        time.sleep(5)

                elif action_key == 'scroll':
                    if not args.text_only:
                        exec_action_scroll(info, web_eles, driver_task, args, None)
                    else:
                        exec_action_scroll(info, None, driver_task, args, obs_info)

                elif action_key == 'goback':
                    driver_task.back()
                    time.sleep(2)

                elif action_key == 'google':
                    driver_task.get('https://www.google.com/')
                    time.sleep(2)

                # 這個代表結束
                elif action_key == 'answer':
                    logging.info(info['content'])
                    logging.info('finish!!')
                    break

                else:
                    raise NotImplementedError
                fail_obs = ""
            except Exception as e:
                logging.error('driver error info:')
                logging.error(e)
                if 'element click intercepted' not in str(e):
                    fail_obs = "The action you have chosen cannot be exected. Please double-check if you have selected the wrong Numerical Label or Action or Action format. Then provide the revised Thought and Action."
                else:
                    fail_obs = ""
                time.sleep(2)


        # print_message(messages, task_dir)
        driver_task.quit()
        logging.info(f'Total cost: {accumulate_prompt_token / 1000 * 0.01 + accumulate_completion_token / 1000 * 0.03}')


if __name__ == '__main__':
    main()
    print('End of process')
