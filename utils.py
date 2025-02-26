import fitz  # PyMuPDF required
import os
from markitdown import MarkItDown
from openai import OpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
import chromadb
import os
from PIL import Image
from IPython.display import Image as IPImage, display
import re
import webbrowser


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="image_desc_vectordb")
collection = chroma_client.get_or_create_collection("image_desc")

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


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


def create_vectordb(output_folder):
    images_path = output_folder  # Since images and code are in the same folder
    code_path = output_folder

    client = OpenAI(api_key=OPENAI_API_KEY)
    md = MarkItDown(llm_client=client, llm_model="gpt-4o")

    for filename in os.listdir(images_path):
        if filename.startswith("image_") and filename.endswith((".jpeg", ".png", ".jpg")):
            number_part = filename.split("_")[1].split(".")[0]
            code_filename = f"code_{number_part}.txt"
            code_filepath = os.path.join(code_path, code_filename)

            if os.path.exists(code_filepath):
                with open(code_filepath, "r") as f:
                    code = f.read()

                # Generate the image description using MarkItDown
                desc = md.convert(os.path.join(images_path, filename))
                image_content = desc.text_content

                query = f"CODE: {code}\nCORRESPONDING FLOWCHART IMAGE: {image_content}"
                embedding = embeddings.embed_query(query)

                # Check if the embedding already exists for this image ID
                existing = collection.get(ids=[filename])
                if existing and existing.get("ids") and existing["ids"][0]:
                    print(f"Embedding already exists for {filename}. Skipping addition.")
                else:
                    collection.add(
                        ids=[filename],
                        documents=[query],
                        embeddings=[embedding]
                    )
                    print(f"Added embedding for {filename}")
            else:
                print(f"Warning: No matching code file found for {filename}")

    print("Vector database populated successfully!")


def query_flowchart(query, output_folder):
    query_embedding = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1  
    )

    if results["ids"] and results["ids"][0]:
        image_filename = results["ids"][0][0]
        document_text = results["documents"][0][0]  # Retrieve the stored document

        image_path = os.path.join(output_folder, image_filename)
        if os.path.exists(image_path):
            display(IPImage(filename=image_path))
            print("Image Description and Code:")
            print(document_text)
        else:
            print(f"Image not found: {image_path}")
    else:
        print("No matching flowchart found.")


def query_flowchart_for_prompt(query, output_folder):
    query_embedding = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1  
    )

    if results["ids"] and results["ids"][0]:
        image_filename = results["ids"][0][0]
        document_text = results["documents"][0][0]  # Retrieve stored document

        # Construct the LLM prompt
        llm_prompt = f"""
        Below is a flowchart extracted from a document along with its corresponding code.

        **Flowchart Description:**
        {document_text}

        **Corresponding Code:**
        {document_text.split("CODE:")[1] if "CODE:" in document_text else "No code found"}

        
        Generate a new flowchart that represents the above logic using MermaidJS syntax. Output MermaidJS syntax only.
        """
        return llm_prompt


def generate_flowchart(llm_prompt):
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    api_key = OPENAI_API_KEY
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": llm_prompt}],
        max_tokens=500
    )
    flowchart_code = response.choices[0].message.content.strip()

    # Remove possible backticks and language tags (e.g., ```mermaid ... ```)
    flowchart_code = re.sub(r"^```(?:mermaid)?\n?|```$", "", flowchart_code, flags=re.MULTILINE).strip()

    print()
    print("----FLOWCHART-CODE----")

    print(flowchart_code)
    print("----------END-----------")

    return flowchart_code




def render_mermaid(flowchart_code):
    mermaid_script = f"""
    <html>
    <head>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
    </head>
    <body>
    <div class="mermaid">
    {flowchart_code}
        
    </div>
    </body>
    </html>
    """
    with open("generated_flowchart.html", "w") as f:
        f.write(mermaid_script)
    print("Flowchart saved as 'generated_flowchart.html'. Open it in a browser to view.")




def save_image(image_bytes, output_folder, filename):
    with open(os.path.join(output_folder, filename), "wb") as f:
        f.write(image_bytes)

def open_pdf(pdf_path):
    abs_path = os.path.abspath(pdf_path)  # Get absolute path
    webbrowser.open(f'file://{abs_path}')  # Open in default browser






