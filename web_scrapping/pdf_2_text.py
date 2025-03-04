import fitz
import os

def extract_text_from_pdf(pdf_path, output_dir):
    """
    Extracts text from a PDF and saves it to a text file with the same name.

    Args:
        pdf_path (str): Path to the PDF file.
        output_dir (str): Directory to save the extracted text file.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ''
        for page in doc:
            text += page.get_text() + "\n"
        text = text.strip()

        # Create output file path
        pdf_filename = os.path.basename(pdf_path)
        text_filename = os.path.splitext(pdf_filename)[0] + ".txt"
        output_path = os.path.join(output_dir, text_filename)

        # Save text to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"Extracted text from {pdf_filename} and saved to {text_filename}")

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

def process_pdfs_in_directory(pdf_dir, output_dir):
    """
    Processes all PDFs in a directory and extracts text.

    Args:
        pdf_dir (str): Directory containing PDF files.
        output_dir (str): Directory to save extracted text files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            extract_text_from_pdf(pdf_path, output_dir)

if __name__ == "__main__":
    pdf_directory = "/home/gurupriyan/Desktop/Internship Assignment/Lexiscope/Backend/supreme_court_pdfs"  
    output_directory = "/home/gurupriyan/Desktop/Internship Assignment/Lexiscope/Backend/supreme_court_texts" 
    process_pdfs_in_directory(pdf_directory, output_directory)