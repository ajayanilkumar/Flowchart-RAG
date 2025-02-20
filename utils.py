import fitz  # PyMuPDF required
import os
from markitdown import MarkItDown
from openai import OpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
import chromadb
import os
from PIL import Image
from IPython.display import Image as IPImage, display

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="image_desc_vectordb")
collection = chroma_client.get_or_create_collection("image_desc")

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# def extract_images_and_code(pdf_path, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     pdf_document = fitz.open(pdf_path)
#     global_image_index = 1
#     text_accumulator = ""

#     for page_num in range(len(pdf_document)):
#         page = pdf_document[page_num]
#         # Get page blocks as a dict and sort them in reading order (top-to-bottom)
#         page_dict = page.get_text("dict")
#         blocks = page_dict["blocks"]
#         blocks.sort(key=lambda b: b["bbox"][1])
        
#         for block in blocks:
#             if block["type"] == 0:  # Text block
#                 # Extract text by concatenating all spans from the block
#                 block_text = ""
#                 for line in block.get("lines", []):
#                     for span in line.get("spans", []):
#                         block_text += span.get("text", "") + " "
#                     block_text += "\n"
#                 text_accumulator += block_text.strip() + "\n"
#             elif block["type"] == 1:  # Image block
#                 # When an image is encountered, save the accumulated text as the corresponding code file.
#                 code_filename = os.path.join(output_folder, f"code_{global_image_index}.txt")
#                 with open(code_filename, "w") as cf:
#                     cf.write(text_accumulator.strip())
#                 # Reset the accumulator for the next snippet.
#                 text_accumulator = ""
                
#                 # Extract and save the image with the corresponding number
#                 xref = block.get("xref")
#                 if xref:
#                     base_image = pdf_document.extract_image(xref)
#                     image_ext = base_image["ext"]
#                     image_filename = os.path.join(output_folder, f"image_{global_image_index}.{image_ext}")
#                     with open(image_filename, "wb") as img_file:
#                         img_file.write(base_image["image"])
#                     print(f"Saved image: {image_filename} and code: {code_filename}")
#                 else:
#                     print("Warning: Image block without xref found.")
                
#                 global_image_index += 1

#     # Save any remaining text at the end as a trailing code snippet
#     if text_accumulator.strip():
#         code_filename = os.path.join(output_folder, f"code_{global_image_index}.txt")
#         with open(code_filename, "w") as cf:
#             cf.write(text_accumulator.strip())
#         print(f"Saved trailing code snippet as: {code_filename}")
    
#     print(f"Extraction complete! Files saved in: {output_folder}")

# # Example usage:
# pdf_path = "Codes_and_Flowcharts_Dataset.pdf"
# output_folder = "extracted_data"
# extract_images_and_code(pdf_path, output_folder)


def extract_images_and_code(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    global_image_index = 1
    text_accumulator = ""

    for page_num, page in enumerate(pdf_document):
        # Extract text
        page_dict = page.get_text("dict")
        blocks = page_dict["blocks"]
        blocks.sort(key=lambda b: b["bbox"][1])  # Sort blocks by their Y-coordinate

        for block in blocks:
            if block["type"] == 0:  # Text block
                block_text = " ".join(span["text"] for line in block["lines"] for span in line["spans"])
                text_accumulator += block_text.strip() + "\n"

        # Extract images using `get_images(full=True)`
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]  # Get the xref of the image
            base_image = pdf_document.extract_image(xref)
            if base_image:
                image_ext = base_image["ext"]
                image_filename = os.path.join(output_folder, f"image_{global_image_index}.{image_ext}")
                with open(image_filename, "wb") as img_file:
                    img_file.write(base_image["image"])
                print(f"Saved image: {image_filename}")

                # Save accumulated text before each image
                if text_accumulator.strip():
                    code_filename = os.path.join(output_folder, f"code_{global_image_index}.txt")
                    with open(code_filename, "w") as cf:
                        cf.write(text_accumulator.strip())
                    print(f"Saved corresponding code: {code_filename}")
                    text_accumulator = ""  # Reset accumulator
                
                global_image_index += 1

    # Save remaining text if any
    if text_accumulator.strip():
        code_filename = os.path.join(output_folder, f"code_{global_image_index}.txt")
        with open(code_filename, "w") as cf:
            cf.write(text_accumulator.strip())
        print(f"Saved trailing code snippet: {code_filename}")

    print(f"Extraction complete! Files saved in: {output_folder}")

# # Example usage:
# pdf_path = "Codes_and_Flowcharts_Dataset.pdf"
# output_folder = "extracted_data"
# extract_images_and_code(pdf_path, output_folder)

def create_vectordb(images_path, code_path):
    client = OpenAI(api_key=OPENAI_API_KEY)
    md = MarkItDown(llm_client=client, llm_model="gpt-4o")

    for filename in os.listdir(images_path):
        # Only process image files (assuming JPEG here)
        if filename.endswith(".jpeg"):
            image_path = os.path.join(images_path, filename)
            
            # Extract the numeric part from the filename
            # Assuming the filename format is "image_1.jpeg", "image_2.jpeg", etc.
            if filename.startswith("image_"):
                number_part = filename.split("_")[1].split(".")[0]
                code_filename = f"code_{number_part}.txt"
            else:
                # Fallback to using the same base name if pattern doesn't match
                base_name = os.path.splitext(filename)[0]
                code_filename = base_name + ".txt"
            
            code_filepath = os.path.join(code_path, code_filename)

            if os.path.exists(code_filepath):
                with open(code_filepath, "r") as f:
                    code = f.read()

                # Process the image and create a textual description using MarkItDown
                desc = md.convert(image_path)
                image_content = desc.text_content

                # Combine code and image description for the query
                query = f"CODE: {code}\nCORRESPONDING FLOWCHART IMAGE: {image_content}"
                
                print(query,"/n")


                # Generate the embedding using your embeddings object
                embedding = embeddings.embed_query(query)

                # Add the document to your vector database collection
                collection.add(
                    ids=[filename],       # Use the image filename as the document ID
                    documents=[query],      # Store the combined query for retrieval
                    embeddings=[embedding]
                )
            else:
                print(f"Warning: No matching code file found for {filename}")

    print("Vector database populated successfully!")


def query_flowchart(query, output_folder):
    
    query_embedding = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1  
    )
    image_path = results['ids'][0][0]

    image_path = os.path.join(output_folder, image_path)
    if os.path.exists(image_path) and image_path.lower().endswith('.jpeg'):
        display(IPImage(filename=image_path))
    else:
        print(f"Image not found : {image_path}")



def extract_code_and_images(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    pdf_document = fitz.open(pdf_path)

    image_index = 1
    code_text = ""
    code_filename = None

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        blocks = page.get_text("blocks")

        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block

            if block_type == 0:  # Text block
                if code_filename is None:
                    code_filename = os.path.join(output_folder, f"code_{image_index}.txt")
                code_text += text + "\n"

            elif block_type == 1:  # Image block
                image_xrefs = page.get_image_xrefs() #Get all the image xrefs on the page
                for xref in image_xrefs: #Iterate through each xref
                    image = pdf_document.extract_image(xref) #Extract the image using the xref
                    image_bytes = image["image"]
                    image_ext = image["ext"]
                    image_filename = os.path.join(output_folder, f"image_{image_index}.{image_ext}")
                    with open(image_filename, "wb") as img_file:
                        img_file.write(image_bytes)

                    if code_filename:
                        with open(code_filename, "w") as code_file:
                            code_file.write(code_text)
                        code_text = ""
                        code_filename = None
                    image_index += 1

    print(f"Images and code extracted and saved to {output_folder}")
    return


def save_image(image_bytes, output_folder, filename):
    with open(os.path.join(output_folder, filename), "wb") as f:
        f.write(image_bytes)
