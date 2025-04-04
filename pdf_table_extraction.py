import os
import re
import pandas as pd
import PyPDF2

def extract_tables_from_text(text):
    """Extract tables from text based on line patterns"""
    tables = []
    current_table = []
    lines = text.split('\n')
    
    # Simple heuristic: consecutive lines with similar structure might be table rows
    in_table = False
    
    for i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            if in_table and current_table:
                tables.append(process_text_table(current_table))
                current_table = []
                in_table = False
            continue
        
        # Check if this line looks like a table row (has multiple whitespace-separated items)
        items = [item.strip() for item in re.split(r'\s{2,}', line) if item.strip()]
        
        if len(items) >= 2:  # Potential table row with at least 2 cells
            if not in_table:
                in_table = True
            current_table.append(items)
        else:
            if in_table and current_table:
                tables.append(process_text_table(current_table))
                current_table = []
                in_table = False
    
    # Don't forget the last table if we ended in_table mode
    if in_table and current_table:
        tables.append(process_text_table(current_table))
    
    return tables

def process_text_table(rows):
    """Process rows extracted from text to make a uniform table"""
    if not rows:
        return []
    
    # Find the maximum number of columns
    max_cols = max(len(row) for row in rows)
    
    # Pad rows to have the same number of columns
    for i in range(len(rows)):
        while len(rows[i]) < max_cols:
            rows[i].append('')
    
    return rows

def save_to_excel(tables, output_path):
    """Save extracted tables to Excel file"""
    if not tables:
        print("No tables found to save")
        return
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for i, table_data in enumerate(tables):
            # Convert table data to DataFrame
            df = pd.DataFrame(table_data)
            # Clean up DataFrame (remove empty rows/columns)
            df = df.replace('', None)
            df = df.dropna(how='all', axis=0)  # Drop empty rows
            df = df.dropna(how='all', axis=1)  # Drop empty columns
            
            # Write to Excel
            sheet_name = f"Table_{i+1}"
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets[sheet_name]
            for j, col in enumerate(df.columns):
                max_length = 0
                for row in df.index:
                    cell_value = str(df.iloc[row, j])
                    if cell_value:
                        max_length = max(max_length, len(cell_value))
                worksheet.column_dimensions[chr(65 + j)].width = max_length + 2

def process_pdf_simple(pdf_path, output_excel_path):
    """Process a PDF file using PyPDF2 for simple table extraction"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
        
        tables = extract_tables_from_text(full_text)
        save_to_excel(tables, output_excel_path)
        print(f"Processed using simple method - found {len(tables)} potential tables")
        return tables
    except Exception as e:
        print(f"Error in simple processing: {str(e)}")
        return []

def main():
    """Main function to run the script"""
    input_dir = "input_pdf"
    output_dir = "Output_excel"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all PDFs in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            excel_path = os.path.join(output_dir, filename.replace(".pdf", ".xlsx"))
            
            print(f"Processing {filename}...")
            try:
                process_pdf_simple(pdf_path, excel_path)
                print(f"Saved extracted tables to {excel_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()