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


def extract_images_and_code(pdf_path, output_folder):
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
                if code_filename is None: #check if this is the first line of code for this image
                    code_filename = os.path.join(output_folder, f"code_{image_index}.txt")
                code_text += text + "\n"  # Accumulate ALL text as code

            elif block_type == 1:  # Image block
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_filename = os.path.join(output_folder, f"image_{image_index}.{image_ext}")
                    with open(image_filename, "wb") as img_file:
                        img_file.write(image_bytes)

                    if code_filename:  # Save code if associated with the image
                        with open(code_filename, "w") as code_file:
                            code_file.write(code_text)
                        code_text = ""  # Reset for the next code block
                        code_filename = None
                    image_index += 1

    print(f"Images and code extracted and saved to {output_folder}")
    return


def create_vectordb(images_path, code_path):
    client = OpenAI(api_key=OPENAI_API_KEY)
    md = MarkItDown(llm_client=client, llm_model="gpt-4o")

    for filename in os.listdir(images_path):
        if filename.endswith(".jpeg"): 
            image_path = os.path.join(images_path, filename)
            base_name = os.path.splitext(filename)[0]  # Filename without extension

            # 1. Find Corresponding Code File:
            code_filename = base_name + ".txt" 
            code_filepath = os.path.join(code_path, code_filename)

            if os.path.exists(code_filepath):
                with open(code_filepath, "r") as f:
                    code = f.read()

                # 2. Process Image and Create Query:
                desc = md.convert(image_path)
                image_content = desc.text_content

                query = f"CODE: {code}\nCORRESPONDING FLOWCHART IMAGE: {image_content}"

                # embedding = client.embeddings.create(input=[query], model="text-embedding-ada-002")['data'][0]['embedding'] #Use OpenAI library for embedding

                
                embedding = embeddings.embed_query(query)

                collection.add(  # Assuming 'collection' is your vector database object
                    ids=[filename], #Use Filename as ID
                    documents=[query],  # Store the combined query for retrieval
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
    doc = fitz.open(pdf_path)
    all_data = []
    current_code = ""

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text_blocks = page.get_text("blocks")

        for block in text_blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            if block_type == 0:  # Text block (code)
                current_code += text + "\n"

            elif block_type == 1:  # Image block
                image_x0, image_y0, image_x1, image_y1, *_ = block
                image_xref = page.get_image_xrefs()

                for xref in image_xref:
                    rect = page.get_image_rects(xref)
                    if rect.x0 == image_x0 and rect.y0 == image_y0 and rect.x1 == image_x1 and rect.y1 == image_y1:
                        image = doc.extract_image(xref)
                        image_bytes = image["image"]
                        all_data.append((current_code.strip(), image_bytes))
                        current_code = ""  # Reset code after associating with image
                        break  # Important: Exit inner loop after finding the image

        # Check for code at the end of the current page
        if current_code:
            next_page_num = page_num + 1
            if next_page_num < doc.page_count:
                next_page = doc[next_page_num]
                next_page_images = next_page.get_images(full=True)  # Get all images

                # Filter small images based on their size (bytes)
                filtered_images = []
                for image_info in next_page_images:
                    xref = image_info[0]
                    image = doc.extract_image(xref)
                    image_bytes = image["image"]
                    if len(image_bytes) > 1000:  # Example: Keep images larger than 1KB
                        filtered_images.append(image_info)

                if filtered_images:
                    next_image_xref = filtered_images[0][0] # take the first image
                    next_image = doc.extract_image(next_image_xref)
                    next_image_bytes = next_image["image"]
                    all_data.append((current_code.strip(), next_image_bytes))
                    current_code = ""
                else:
                    all_data.append((current_code.strip(), None))
                    current_code = ""

            else:
                all_data.append((current_code.strip(), None))
                current_code = ""

    return all_data



def save_image(image_bytes, output_folder, filename):
    with open(os.path.join(output_folder, filename), "wb") as f:
        f.write(image_bytes)

# Example usage:
pdf_path = "Codes_and_Flowcharts_Dataset.pdf"
output_folder = "extracted_data"
# import os
# os.makedirs(output_folder, exist_ok=True)
# extracted_data = extract_code_and_images(pdf_path, output_folder)

# for i, (code, image_bytes) in enumerate(extracted_data):
#     if image_bytes:
#         save_image(image_bytes, output_folder, f"image_{i}.jpeg")  # Save image
#     with open(os.path.join(output_folder,f"code_{i}.txt"),"w") as f:
#         f.write(code) # save code

#     print(f"--- Data Point {i+1} ---")
#     print("Code:", code)
#     if image_bytes:
#       print("Image saved as image_{i}.jpeg")
#     else:
#       print("No associated Image")
#     print("-" * 20)


extract_images_and_code_from_pdf(pdf_path, output_folder)