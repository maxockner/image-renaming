import argparse
import openai
import os 
import replicate
from PIL import Image
    
openai.api_key = os.getenv("OPENAI_API_KEY")

REPLICATE_IMAGE_CAPTION_MODEL = "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608"
OPENAI_GPT_MODEL="gpt-3.5-turbo"

def get_image_caption(image_path, context=None):
    input = {}
    input["image"] = open(image_path, "rb")
    if context is not None:
        input["context"] = context
    return replicate.run(REPLICATE_IMAGE_CAPTION_MODEL, input=input)

GPT_SYSTEM_PROMPT = '''
You will be given a description of an image, and you will output a plausible and descriptive file name for the image. A valid filename is allow to have alphanumeric characters and dashes. 
Do not include spaces or other special characters! Do not include a file extension. 
Example:
---- begin example
description: dog chasing squirrels in a park
filename: dog-chasing-squirrel
---- end example
'''

def get_image_filename(caption):
    response = openai.ChatCompletion.create(
        model=OPENAI_GPT_MODEL,
        messages=[
            {
            "role": "system",
            "content": GPT_SYSTEM_PROMPT,
            },
            {
            "role": "user",
            "content": f"description: {caption}\nfilename: "
            }
        ],
        temperature=0,
        max_tokens=256,
    )
    return response.choices[0].message.content

def get_image_type(filename):
    try:
        img = Image.open(filename)
        img.verify()
        return img.format
    except (IOError, SyntaxError):
        return None
    
def get_image_caption_and_filename(image_path):
    ext = os.path.splitext(image_path)[1]
    # image_type = get_image_type(image_path)
    caption = get_image_caption(image_path)
    filename = get_image_filename(caption)
    return {
        "filename": f"{filename}{ext}",
        "caption": caption,
    }

image_path = '/Users/maxockner/Desktop/Screen Shot 2023-01-15 at 12.50.05 AM.png'
# print(get_image_caption_and_filename(image_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', help='path to your file')

    args = parser.parse_args()
    print("input file:", args.filename)
    print(get_image_caption_and_filename(args.filename))

if __name__ == '__main__':
    main()