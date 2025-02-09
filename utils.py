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


def extract_images_from_pdf(pdf_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"] 
            image_filename = os.path.join(output_folder, f"page{page_num+1}_img{img_index+1}.{image_ext}")
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)
    
    print(f"Images extracted and saved to {output_folder}")

def create_vectordb(images_path):

    client = OpenAI(api_key=OPENAI_API_KEY)
    md = MarkItDown(llm_client=client, llm_model="gpt-4o")
    for path in os.listdir(images_path):
        if path.endswith(".jpeg"):
            file_path = f"{images_path}/{path}"
            desc = md.convert(file_path)
            content = desc.text_content

            embedding = embeddings.embed_query(content)

            collection.add(
                ids=[path],  
                documents=[content],  
                embeddings=[embedding]  
            )

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