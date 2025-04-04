import os
import re
import pandas as pd
import PyPDF2
import pdfplumber
import numpy as np
from collections import defaultdict

class PDFTableExtractor:
    def __init__(self):
        """Initialize the PDF Table Extractor with various extraction methods"""
        self.extraction_methods = {
            "text_pattern": self.extract_tables_from_text,
            "pdfplumber": self.extract_tables_with_pdfplumber
        }
    
    def extract_tables_from_text(self, text):
        """Extract tables from text based on line patterns"""
        tables = []
        current_table = []
        lines = text.split('\n')
        
        # Simple heuristic: consecutive lines with similar structure might be table rows
        in_table = False
        table_line_pattern = None
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                if in_table and current_table:
                    tables.append(self.process_text_table(current_table))
                    current_table = []
                    in_table = False
                    table_line_pattern = None
                continue
            
            # Check if this line looks like a table row (has multiple whitespace-separated items)
            # We'll also look for date patterns, monetary values, etc. commonly found in tables
            items = [item.strip() for item in re.split(r'\s{2,}', line) if item.strip()]
            
            # Various heuristics to detect table rows
            has_multiple_columns = len(items) >= 2
            has_date_pattern = bool(re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', line))
            has_monetary_value = bool(re.search(r'\d+,\d+\.\d{2}', line) or re.search(r'\d+\.\d{2}', line))
            has_debits_credits = bool(re.search(r'Dr|Cr', line))
            
            # Enhanced table detection logic
            is_table_row = has_multiple_columns or has_date_pattern or has_monetary_value or has_debits_credits
            
            if is_table_row:
                # If this is a new table, establish a pattern
                if not in_table:
                    in_table = True
                    # Create a simple pattern signature of the line
                    table_line_pattern = self._get_line_pattern(line)
                
                # If we're in a table, check if this row follows a similar pattern to maintain consistency
                if in_table:
                    current_pattern = self._get_line_pattern(line)
                    pattern_similarity = self._pattern_similarity(table_line_pattern, current_pattern)
                    
                    # If patterns are similar, add to current table
                    if pattern_similarity > 0.5:  # Threshold for similarity
                        current_table.append(items)
                    else:
                        # If pattern is too different, close the current table and start a new one
                        if current_table:
                            tables.append(self.process_text_table(current_table))
                            current_table = [items]
                            table_line_pattern = current_pattern
                        else:
                            current_table.append(items)
                            table_line_pattern = current_pattern
            else:
                # Not a table row, close current table if it exists
                if in_table and current_table:
                    tables.append(self.process_text_table(current_table))
                    current_table = []
                    in_table = False
                    table_line_pattern = None
        
        # Don't forget the last table if we ended in_table mode
        if in_table and current_table:
            tables.append(self.process_text_table(current_table))
        
        return tables
    
    def _get_line_pattern(self, line):
        """Create a simple pattern signature of a line to help identify table structures"""
        # Replace digits with 'D', letters with 'L', spaces with 'S', etc.
        pattern = re.sub(r'\d+', 'D', line)
        pattern = re.sub(r'[a-zA-Z]+', 'L', pattern)
        pattern = re.sub(r'\s+', 'S', pattern)
        pattern = re.sub(r'[^\w\s]', 'P', pattern)  # P for punctuation
        return pattern
    
    def _pattern_similarity(self, pattern1, pattern2):
        """Calculate similarity between two line patterns"""
        # Simple similarity metric: proportion of matching characters
        if not pattern1 or not pattern2:
            return 0
        
        # Get the length of the shorter pattern
        min_len = min(len(pattern1), len(pattern2))
        
        # Count matching characters
        matches = sum(1 for i in range(min_len) if pattern1[i] == pattern2[i])
        
        # Return similarity ratio
        return matches / min_len
    
    def process_text_table(self, rows):
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
    
    def extract_tables_with_pdfplumber(self, pdf_path):
        """Extract tables using PDFPlumber which works well with bordered tables"""
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Extract tables with explicit borders
                    extracted_tables = page.extract_tables()
                    
                    for table in extracted_tables:
                        if table:
                            tables.append(table)
                    
                    # Try to find tables without borders using text analysis
                    if not extracted_tables:
                        text_tables = self._extract_implicit_tables_pdfplumber(page)
                        tables.extend(text_tables)
                        
                    # Additional method: Try to find tables by analyzing word positions
                    position_tables = self._extract_tables_by_position(page)
                    tables.extend(position_tables)
        except Exception as e:
            print(f"PDFPlumber extraction error: {str(e)}")
        
        return tables
    
    def _extract_implicit_tables_pdfplumber(self, page):
        """Extract tables that don't have explicit borders using spacing analysis"""
        tables = []
        
        try:
            # Get all text with position information
            words = page.extract_words()
            
            if not words:
                return []
            
            # Group words by their y-position (approximate rows)
            rows = defaultdict(list)
            for word in words:
                y_pos = round(word['top'], 0)  # Round to nearest pixel
                rows[y_pos].append(word)
            
            # Sort rows by y-position
            sorted_rows = [rows[y] for y in sorted(rows.keys())]
            
            # Analyze horizontal positions to detect columns
            all_x_positions = []
            for row in sorted_rows:
                for word in row:
                    all_x_positions.append(word['x0'])
            
            # Find clustering of x positions to identify potential columns
            if all_x_positions:
                # Simple clustering by binning
                hist, bin_edges = np.histogram(all_x_positions, bins=min(20, len(all_x_positions)//5 + 1))
                potential_cols = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges)-1) if hist[i] > len(sorted_rows) * 0.1]
                
                # If we found potential columns, reconstruct the table
                if potential_cols:
                    table_data = []
                    for row in sorted_rows:
                        table_row = [''] * len(potential_cols)
                        
                        for word in sorted(row, key=lambda w: w['x0']):
                            # Find closest column
                            col_idx = min(range(len(potential_cols)), 
                                        key=lambda i: abs(potential_cols[i] - word['x0']))
                            
                            # Append text to that column
                            if table_row[col_idx]:
                                table_row[col_idx] += ' ' + word['text']
                            else:
                                table_row[col_idx] = word['text']
                        
                        table_data.append(table_row)
                    
                    # Check if the table has enough rows and columns to be considered a table
                    if len(table_data) >= 3 and any(len(set(col)) > 1 for col in zip(*table_data)):
                        tables.append(table_data)
        
        except Exception as e:
            print(f"Implicit table extraction error: {str(e)}")
        
        return tables
    
    def _extract_tables_by_position(self, page):
        """Extract tables by analyzing word positions and alignments"""
        tables = []
        
        try:
            # Extract words with their positions
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            
            if not words:
                return []
            
            # Find lines by grouping words with similar y-positions
            y_tolerance = 5  # Pixels tolerance for considering words on the same line
            lines = defaultdict(list)
            
            for word in words:
                # Round y position to identify lines
                y_pos = round(word['top'] / y_tolerance) * y_tolerance
                lines[y_pos].append(word)
            
            # Sort lines by y-position (top to bottom)
            sorted_lines = [sorted(lines[y], key=lambda w: w['x0']) for y in sorted(lines.keys())]
            
            # Find potential tables by analyzing consistent x-positions across multiple lines
            if len(sorted_lines) >= 3:  # Need at least 3 lines to form a table
                # Analyze x-positions across all lines to find column boundaries
                x_positions = []
                for line in sorted_lines:
                    for word in line:
                        x_positions.append(word['x0'])
                
                # Use clustering to identify column positions
                if x_positions:
                    # Find frequently occurring x-positions
                    x_pos_counts = defaultdict(int)
                    for x in x_positions:
                        # Round x-position to handle slight variations
                        rounded_x = round(x / 5) * 5
                        x_pos_counts[rounded_x] += 1
                    
                    # Get column positions that appear in multiple lines
                    potential_columns = [x for x, count in x_pos_counts.items() 
                                        if count >= len(sorted_lines) * 0.2]  # Column appears in at least 20% of lines
                    
                    # Sort column positions
                    potential_columns.sort()
                    
                    if len(potential_columns) >= 2:  # Need at least 2 columns
                        # Create table by assigning words to columns
                        table_data = []
                        
                        # Process each line
                        for line in sorted_lines:
                            if not line:
                                continue
                                
                            # Create empty row
                            row_data = [''] * (len(potential_columns))
                            
                            # Assign words to columns
                            for word in line:
                                # Find the closest column position
                                col_idx = 0
                                min_dist = float('inf')
                                
                                for i, col_pos in enumerate(potential_columns):
                                    dist = abs(word['x0'] - col_pos)
                                    if dist < min_dist:
                                        min_dist = dist
                                        col_idx = i
                                
                                # Add word to appropriate column
                                if row_data[col_idx]:
                                    row_data[col_idx] += ' ' + word['text']
                                else:
                                    row_data[col_idx] = word['text']
                            
                            table_data.append(row_data)
                        
                        # Check if this looks like a valid table (contains actual data)
                        if self._is_valid_table(table_data):
                            tables.append(table_data)
        
        except Exception as e:
            print(f"Position-based table extraction error: {str(e)}")
        
        return tables
    
    def _is_valid_table(self, table_data):
        """Check if the extracted data looks like a valid table"""
        if not table_data or len(table_data) < 2:
            return False
        
        # Check if we have enough non-empty cells
        total_cells = len(table_data) * len(table_data[0])
        non_empty_cells = sum(1 for row in table_data for cell in row if cell.strip())
        
        # At least 30% of cells should be non-empty
        if non_empty_cells / total_cells < 0.3:
            return False
        
        # Check for at least some columns with consistent content
        consistent_cols = 0
        for col_idx in range(len(table_data[0])):
            col_values = [row[col_idx] for row in table_data if col_idx < len(row)]
            non_empty_values = [v for v in col_values if v.strip()]
            
            # Check if column has consistent formatting
            if len(non_empty_values) >= 3:
                # Check for numeric column
                numeric_values = [v for v in non_empty_values if re.search(r'\d+\.?\d*', v)]
                if len(numeric_values) >= len(non_empty_values) * 0.7:
                    consistent_cols += 1
                
                # Check for date column
                date_values = [v for v in non_empty_values if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', v)]
                if len(date_values) >= len(non_empty_values) * 0.7:
                    consistent_cols += 1
        
        return consistent_cols > 0
    
    def extract_bank_statement_data(self, pdf_path):
        """Special method for extracting bank statement data which is often in a specific format"""
        tables = []
        transaction_data = []
        
        try:
            # Extract text with PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() + "\n"
            
            # Method 1: Look for transaction patterns in bank statements
            # This pattern matches date, description, amount entries common in bank statements
            transaction_pattern = r'(\d{2}[-/]\w{3}[-/]\d{4}|\d{2}[-/]\d{2}[-/]\d{4}|[A-Z][a-z]{2}[-/]\d{2}[-/]\d{4})\s+(.+?)\s+(\d{1,3}(?:,\d{3})*\.\d{2})\s+((?:\d{1,3}(?:,\d{3})*\.\d{2})?Dr|Cr)?'
            
            matches = re.finditer(transaction_pattern, full_text)
            
            for match in matches:
                date, description, amount, balance = match.groups()
                transaction_data.append([date, description.strip(), amount, balance if balance else ""])
            
            # Method 2: Look for specific patterns in bank statements
            # First, split by lines
            lines = full_text.split('\n')
            
            # Look for table headers or specific date formats
            date_patterns = [
                r'\d{2}[-/]\w{3}[-/]\d{4}',  # 01-Apr-2022
                r'\d{2}[-/]\d{2}[-/]\d{4}',   # 01-04-2022
                r'\d{2}[-/]\d{2}[-/]\d{2}'    # 01-04-22
            ]
            
            # Find transaction lines by looking for date patterns at the beginning
            possible_transaction_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line starts with a date pattern
                for pattern in date_patterns:
                    if re.match(pattern, line):
                        possible_transaction_lines.append(line)
                        break
            
            # If we found transaction lines, try to parse them
            if possible_transaction_lines:
                # Find common structure in transaction lines
                sample_lines = possible_transaction_lines[:min(10, len(possible_transaction_lines))]
                
                # Try to identify column positions based on space patterns
                column_positions = self._identify_column_positions(sample_lines)
                
                if column_positions:
                    # Parse transaction lines using column positions
                    parsed_transactions = []
                    for line in possible_transaction_lines:
                        transaction = []
                        last_pos = 0
                        
                        for pos in column_positions:
                            if pos <= len(line):
                                cell = line[last_pos:pos].strip()
                                transaction.append(cell)
                                last_pos = pos
                        
                        # Add the last column
                        if last_pos < len(line):
                            transaction.append(line[last_pos:].strip())
                        
                        if transaction:
                            parsed_transactions.append(transaction)
                    
                    if parsed_transactions:
                        # Try to identify column headers
                        headers = self._identify_bank_statement_headers(full_text)
                        
                        # Add headers if found
                        if headers:
                            parsed_transactions.insert(0, headers)
                        
                        tables.append(parsed_transactions)
            
            # If we extracted transaction data from method 1
            if transaction_data:
                tables.append([["Date", "Description", "Amount", "Balance"]] + transaction_data)
        
        except Exception as e:
            print(f"Bank statement extraction error: {str(e)}")
        
        return tables
    
    def _identify_column_positions(self, lines):
        """Identify column positions based on spacing patterns in a set of lines"""
        if not lines:
            return []
        
        # Find potential column boundaries by looking for consistent spaces
        space_positions = defaultdict(int)
        
        for line in lines:
            # Mark positions where there are spaces
            for i, char in enumerate(line):
                if char.isspace():
                    space_positions[i] += 1
        
        # Find positions where spaces occur frequently
        threshold = len(lines) * 0.5  # Space must occur in at least 50% of lines
        frequent_spaces = [pos for pos, count in space_positions.items() if count >= threshold]
        
        # Group adjacent spaces together
        column_boundaries = []
        current_group = []
        
        for pos in sorted(frequent_spaces):
            if not current_group or pos == current_group[-1] + 1:
                current_group.append(pos)
            else:
                # Take the middle of the space group as the column boundary
                if current_group:
                    column_boundaries.append(current_group[len(current_group) // 2])
                current_group = [pos]
        
        # Add the last group
        if current_group:
            column_boundaries.append(current_group[len(current_group) // 2])
        
        return sorted(column_boundaries)
    
    def _identify_bank_statement_headers(self, text):
        """Try to identify column headers in bank statement text"""
        # Common bank statement headers
        common_headers = [
            ["Date", "Description", "Debit", "Credit", "Balance"],
            ["Date", "Particulars", "Withdrawals", "Deposits", "Balance"],
            ["Date", "Description", "Amount", "Balance"],
            ["Date", "Transaction", "Debit", "Credit", "Balance"],
            ["Date", "Narration", "Withdrawal", "Deposit", "Balance"]
        ]
        
        # Look for header patterns in the text
        for headers in common_headers:
            header_pattern = r'\b' + r'\b.*?\b'.join(headers) + r'\b'
            if re.search(header_pattern, text, re.IGNORECASE):
                return headers
        
        # Default headers if none were found
        return ["Date", "Description", "Amount", "Balance"]
    
    def process_pdf(self, pdf_path, output_excel_path):
        """Process a PDF file using multiple methods to extract tables"""
        all_tables = []
        
        # Method 1: Simple text-based extraction
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() + "\n"
            
            text_tables = self.extract_tables_from_text(full_text)
            all_tables.extend(text_tables)
            print(f"Text-based method found {len(text_tables)} tables")
        except Exception as e:
            print(f"Error in text-based extraction: {str(e)}")
        
        # Method 2: PDFPlumber extraction (good for bordered tables)
        try:
            pdfplumber_tables = self.extract_tables_with_pdfplumber(pdf_path)
            all_tables.extend(pdfplumber_tables)
            print(f"PDFPlumber method found {len(pdfplumber_tables)} tables")
        except Exception as e:
            print(f"Error in PDFPlumber extraction: {str(e)}")
        
        # Method 3: Specialized bank statement extraction
        try:
            bank_tables = self.extract_bank_statement_data(pdf_path)
            all_tables.extend(bank_tables)
            print(f"Bank statement method found {len(bank_tables)} tables")
        except Exception as e:
            print(f"Error in bank statement extraction: {str(e)}")
        
        # Remove duplicate tables
        unique_tables = self._remove_duplicate_tables(all_tables)
        print(f"After removing duplicates: {len(unique_tables)} unique tables")
        
        # Save to Excel
        self.save_to_excel(unique_tables, output_excel_path)
        
        return unique_tables
    
    def _remove_duplicate_tables(self, tables):
        """Remove duplicate tables based on content similarity"""
        if not tables:
            return []
        
        unique_tables = []
        table_signatures = []
        
        for table in tables:
            # Create a signature of the table for comparison
            signature = ""
            row_count = min(5, len(table))  # Look at first 5 rows or fewer
            
            for i in range(row_count):
                if i < len(table):
                    row = table[i]
                    # Take first 3 items or fewer from each row
                    col_count = min(3, len(row))
                    for j in range(col_count):
                        if j < len(row):
                            cell = str(row[j])
                            signature += cell[:10]  # First 10 chars of each cell
            
            # Check if we've seen a similar table
            is_duplicate = False
            for existing_sig in table_signatures:
                # Simple similarity metric
                similarity = self._string_similarity(signature, existing_sig)
                if similarity > 0.7:  # 70% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tables.append(table)
                table_signatures.append(signature)
        
        return unique_tables
    
    def _string_similarity(self, s1, s2):
        """Calculate simple string similarity ratio"""
        if not s1 or not s2:
            return 0
        
        # Get the length of the strings
        len_s1, len_s2 = len(s1), len(s2)
        
        # Calculate simple Levenshtein distance
        if len_s1 < len_s2:
            return self._string_similarity(s2, s1)
        
        if len_s2 == 0:
            return 0
        
        # Use only first N characters for efficiency
        max_len = min(100, len_s1, len_s2)
        s1, s2 = s1[:max_len], s2[:max_len]
        
        # Count matching characters
        matches = sum(1 for i in range(len(s2)) if i < len(s1) and s1[i] == s2[i])
        
        return matches / max(len_s1, len_s2)
    
    def save_to_excel(self, tables, output_path):
        """Save extracted tables to Excel file"""
        if not tables:
            print("No tables found to save")
            return
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for i, table_data in enumerate(tables):
                if not table_data:
                    continue
                
                # Convert table data to DataFrame
                df = pd.DataFrame(table_data)
                
                # If first row looks like headers, use it as such
                if i == 0 and len(df) > 1:
                    headers = df.iloc[0].tolist()
                    df = df[1:]
                    df.columns = headers
                
                # Clean up DataFrame
                df = df.replace('', pd.NA)
                df = df.dropna(how='all', axis=0)  # Drop empty rows
                df = df.dropna(how='all', axis=1)  # Drop empty columns
                
                # Write to Excel
                sheet_name = f"Table_{i+1}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Auto-adjust column widths
                worksheet = writer.sheets[sheet_name]
                for j, col in enumerate(df.columns):
                    max_length = 0
                    column_name = str(col)
                    max_length = max(max_length, len(column_name))
                    
                    for k, cell_value in enumerate(df[col]):
                        if cell_value:
                            cell_str = str(cell_value)
                            max_length = max(max_length, len(cell_str))
                    
                    adjusted_width = max_length + 2
                    col_letter = chr(65 + j)
                    if j >= 26:  # For columns beyond Z
                        col_letter = chr(64 + j//26) + chr(65 + j%26)
                    
                    worksheet.column_dimensions[col_letter].width = min(adjusted_width, 50)  # Cap width at 50

def main():
    """Main function to run the script"""
    extractor = PDFTableExtractor()
    
    input_dir = "input_pdf"
    output_dir = "output_excel"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all PDFs in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            excel_path = os.path.join(output_dir, filename.replace(".pdf", ".xlsx"))
            
            print(f"\nProcessing {filename}...")
            try:
                extractor.process_pdf(pdf_path, excel_path)
                print(f"Saved extracted tables to {excel_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()