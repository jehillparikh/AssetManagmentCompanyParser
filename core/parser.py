from core.amcparser import AMCPortfolioParser
import pandas as pd
import re

class ICICIMFParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)


    def _clean_fund_name(self, fund_name):
        """
        Cleans the fund name by removing unwanted characters and normalizing it.
        This method can be overridden in subclasses for specific fund name cleaning logic.
        This is probably not needed for this AMC: ICICI Mutual Fund, but included for consistency.
        """
        # Default implementation: strip whitespace and convert to lowercase
        cleaned_name = re.sub(r'\s+fund.*', ' fund', fund_name, flags=re.IGNORECASE)
        return cleaned_name.strip()    
    
    def _get_fund_name(self, sheet_df):
        """
        Extracts the fund name from the sheet DataFrame.
        This method can be overridden in subclasses for specific fund name extraction logic.
        This is probably not needed for this AMC: ICICI Mutual Fund, but included for consistency.
        """
        # Default implementation: use the first non-empty cell in the first row
        
        return None


    def process_sheet(self, datafile, sheet_name, sheet_df):

        print(f"\n🔍 Processing  → Sheet: {sheet_name}")
        fund_name = self._default_fund_name_extraction(sheet_df)
        fund_name = self._clean_fund_name(fund_name) if fund_name else None

        if fund_name is not None and sheet_name:
                print(f"\n🔍 Processing  → Fund: {fund_name}")

                fund_isin=self._get_fund_isin(fund_name)

                header_row_idx = next(
                        (index for index, row in sheet_df.iterrows() if any("ISIN" in str(val) for val in row.dropna())),
                        None
                )
                if header_row_idx is None:
                        print(f"⚠️ Skipping {sheet_name} (No ISIN header found)")
                        return

                df_clean = pd.read_excel(datafile, sheet_name=sheet_name, skiprows=header_row_idx, dtype=str)
                
                df_clean.columns = df_clean.iloc[0]
                df_clean = df_clean[1:].reset_index(drop=True)

                df_clean = df_clean.loc[:, df_clean.columns.notna()]
                
                # Clean raw column names specifically for ICICI (e.g. hidden tab in ISIN)
                # This should be done BEFORE mapping, on the actual column names from the file.
                df_clean.columns = [col.replace('\tISIN', 'ISIN') for col in df_clean.columns]
                print("Original columns after ISIN cleanup:", df_clean.columns)

                col_mapping = self.column_mapping  #obtain col maping from config to standardize column names
                df_clean=df_clean.rename(columns=col_mapping)
                print("Columns after renaming:", df_clean.columns)
                    
                # Check if 'Coupon' column exists AFTER mapping, if not, add it with default value 0.
                # This is due to the structure of some ICICI AMC files.
                if "Coupon" not in df_clean.columns:
                        # Find a suitable index to insert 'Coupon'. Typically after 'ISIN' or 'Name of Instrument'.
                        # If 'ISIN' exists, insert after it. Otherwise, try after 'Name of Instrument'. Default to 2 or 3 if not found.
                        insert_idx = 3 # Default insert index
                        if 'ISIN' in df_clean.columns:
                            insert_idx = df_clean.columns.get_loc('ISIN') + 1
                        elif 'Name of Instrument' in df_clean.columns:
                             insert_idx = df_clean.columns.get_loc('Name of Instrument') + 1

                        df_clean.insert(insert_idx, 'Coupon', 0) # Insert with default value 0
                        print(f"Inserted 'Coupon' column at index {insert_idx}.")
                else: # If "Coupon" exists from mapping, ensure it's numeric and fill NAs
                    df_clean['Coupon'] = pd.to_numeric(df_clean['Coupon'], errors='coerce').fillna(0)


                #col_names=["Name of Instrument","ISIN", "Coupon" ,"Industry", "Quantity", "Market Value", "% to Net Assets", "Yield", "Yield to call"]
                # Column count check after mapping and potential Coupon insertion
                if len(df_clean.columns) >10: # Assuming 10 is a hard limit for expected columns post-processing for ICICI
                        print(f"⚠️ Skipping {sheet_name} (Too many columns: {len(df_clean.columns)}) probably ESG fund or unexpected format)")
                        return

                df_clean.dropna(subset=["ISIN", "Name of Instrument", "Market Value"], inplace=True)


                #Just a simple logic to determine the type of instrument need to update later TODO

                df_clean[['Yield']] = df_clean[['Yield']].fillna(value=0)
                df_clean['Type'] = df_clean['Yield'].apply(lambda x: 'Debt or related' if x != 0 else 'Equity or Equity related')
                    
                df_clean = df_clean.round(2)
                df_clean["Scheme Name"] = fund_name
                df_clean["AMC"] = self.amc_name
                df_clean["Scheme ISIN"] = fund_isin if fund_isin is not None else None
                

                # Standardize column names for outputs
                df_clean = df_clean[[col for col in self.final_columns if col in df_clean.columns]]

                self.full_data=pd.concat([self.full_data,df_clean],ignore_index=True) if not self.full_data.empty else df_clean
               
# Templates for all other AMC names
class One360Parser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for 360 One Asset Management
        pass


class AdityaBirlaParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Aditya Birla Sun Life Mutual Fund
        pass


class AxisParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)
        self.axis_scheme_map = {}
        self.axis_scheme_map_file = None

    def _get_fund_name(self, sheet_df_ignored): # sheet_df is ignored for Axis
        """
        Retrieves the full fund name for Axis Mutual Fund using the current sheet name (abbreviation)
        by looking it up in a map created from the "Index" sheet of the current datafile.
        Relies on self.current_datafile and self.current_sheet_name set by AMCPortfolioParser.
        """
        datafile = self.current_datafile
        sheet_name_key = self.current_sheet_name

        if sheet_name_key == "Index": # Do not process the "Index" sheet itself as a fund data sheet
            return None

        # Initialize or update scheme map if datafile changed or map not present
        if not self.axis_scheme_map or self.axis_scheme_map_file != datafile:
            try:
                fund_names_df = pd.read_excel(datafile, sheet_name="Index", dtype=str)

                if fund_names_df.empty:
                    print(f"⚠️ 'Index' sheet in {datafile} is empty.")
                    self.axis_scheme_map = {}
                # Assuming 'Short Name' and 'Scheme Name' are the direct column headers in "Index" sheet
                elif 'Short Name' in fund_names_df.columns and 'Scheme Name' in fund_names_df.columns:
                    # Strip whitespace from keys to ensure proper lookup
                    self.axis_scheme_map = dict(zip(fund_names_df['Short Name'].str.strip(), fund_names_df['Scheme Name'].str.strip()))
                else:
                    # Attempt to find header row if direct column names are not found
                    header_row_index = 0
                    found_header = False
                    for i, row in fund_names_df.head().iterrows(): # Check first few rows
                        if 'Short Name' in row.values and 'Scheme Name' in row.values:
                            header_row_index = i
                            found_header = True
                            break
                    if found_header:
                        fund_names_df.columns = fund_names_df.iloc[header_row_index].str.strip() # Strip whitespace from identified header names
                        fund_names_df = fund_names_df[header_row_index+1:].reset_index(drop=True)
                        # After setting new columns, re-check if 'Short Name' and 'Scheme Name' are present
                        if 'Short Name' in fund_names_df.columns and 'Scheme Name' in fund_names_df.columns:
                             self.axis_scheme_map = dict(zip(fund_names_df['Short Name'].str.strip(), fund_names_df['Scheme Name'].str.strip()))
                        else:
                            print(f"⚠️ 'Short Name' or 'Scheme Name' column still not found in 'Index' sheet of {datafile} after attempting to find and set header.")
                            self.axis_scheme_map = {}
                    else:
                        print(f"⚠️ 'Short Name' or 'Scheme Name' column not found in 'Index' sheet of {datafile}, and dynamic header identification failed.")
                        self.axis_scheme_map = {}

                self.axis_scheme_map_file = datafile
            except Exception as e:
                print(f"❌ Error reading or parsing 'Index' sheet from {datafile} for Axis: {e}")
                self.axis_scheme_map = {}
                self.axis_scheme_map_file = None

        return self.axis_scheme_map.get(sheet_name_key)

    def _clean_fund_name(self, fund_name):
        if not fund_name:
            return None
        # Standard cleaning: remove trailing " fund", " scheme" etc. and extra spaces
        cleaned_name = re.sub(r'\s+(fund|scheme).*', '', fund_name, flags=re.IGNORECASE)
        # Additional cleaning for Axis specific patterns if any, e.g. removing (G) or - Direct Plan
        cleaned_name = re.sub(r'\s*\(G\)$', '', cleaned_name, flags=re.IGNORECASE) # Remove (G) suffix
        cleaned_name = re.sub(r'\s*-\s*Direct\s*Plan$', '', cleaned_name, flags=re.IGNORECASE) # Remove - Direct Plan
        return cleaned_name.strip()

    def process_sheet(self, datafile, sheet_name, sheet_df):
        print(f"\n🔍 Processing Sheet for Axis: {sheet_name} in file: {datafile}")

        fund_name_raw = self._get_fund_name(sheet_df) # sheet_df is ignored

        if not fund_name_raw:
            print(f"ℹ️ Skipping sheet '{sheet_name}' for Axis as its name was not found in the 'Index' sheet map or it is the 'Index' sheet itself.")
            return

        fund_name = self._clean_fund_name(fund_name_raw)
        if not fund_name:
            print(f"⚠️ Skipping sheet '{sheet_name}' for Axis due to invalid cleaned fund name.")
            return

        print(f"Found Fund Name for sheet '{sheet_name}': {fund_name}")
        fund_isin = self._get_fund_isin(fund_name)

        header_row_idx = next(
            (index for index, row in sheet_df.iterrows() if any("ISIN" in str(val) for val in row.dropna())),
            None
        )

        if header_row_idx is None:
            print(f"⚠️ Skipping sheet '{sheet_name}' for Axis (No ISIN header found).")
            return

        df_clean = pd.read_excel(datafile, sheet_name=sheet_name, skiprows=header_row_idx, dtype=str)
        if df_clean.empty or df_clean.iloc[0].isnull().all():
            print(f"⚠️ Dataframe is empty or header is all NaN after skiprows for sheet '{sheet_name}'. Skipping.")
            return

        df_clean.columns = df_clean.iloc[0] # Actual headers from file
        df_clean = df_clean[1:].reset_index(drop=True)
        df_clean = df_clean.loc[:, df_clean.columns.notna()]

        # Apply column mapping from config (maps actual file headers to standardized names)
        col_mapping = self.column_mapping
        df_clean = df_clean.rename(columns=col_mapping)

        # Legacy parser dropped a 'row1' column. If such a column exists by that name *after mapping*, drop it.
        # Or, if the first column is consistently unnamed and needs to be dropped, handle that.
        # For now, assume column_mapping handles getting to the desired columns.
        # If a dummy first column is consistently present in Axis files *before* mapping,
        # the mapping should ideally map it to a name like 'dummy_col' and then it can be dropped.
        # Or, if it's unnamed, it might not be picked up by notna() if fully empty.
        # The legacy code `df_clean.columns = ["row1", ...]` then `drop("row1")` implies the *first column read from data* was this dummy.
        # If the first column read from excel (df_clean.iloc[0]) is consistently junk,
        # and it's not easily identifiable by name to map and drop, this needs care.
        # Let's assume for now that column_mapping is comprehensive.
        # The legacy parser assigned 9 specific column names *after* dropping 'row1'.
        # This means the `column_mapping` should result in these 9 columns if possible.

        # Data Cleaning & Type Determination for Axis, aligned with legacy script

        # Ensure essential columns for processing exist after mapping.
        # Add them with appropriate defaults if they are missing.
        if 'Yield' not in df_clean.columns:
            df_clean['Yield'] = 0 # Legacy default before numeric conversion
        if 'Coupon' not in df_clean.columns:
            df_clean['Coupon'] = 0 # Legacy default before numeric conversion
        # 'Type' will be generated based on 'Yield', so no need to add it here if missing from mapping.
        # However, if 'Type' IS mapped and comes from the file, its initial values will be used.
        # The legacy script assigns 'Type' as a column name read from the file.
        # For robustness, if it's not mapped, it will be created by the specific Axis logic.

        # Convert Yield and Coupon to numeric, then fill NA with 0 (as per legacy)
        # This handles cases where Yield/Coupon might be non-numeric or missing.
        df_clean['Yield'] = pd.to_numeric(df_clean['Yield'], errors='coerce').fillna(0)
        df_clean['Coupon'] = pd.to_numeric(df_clean['Coupon'], errors='coerce').fillna(0)

        # Drop rows with NA in essential identifier columns (as per legacy)
        # Ensure these columns exist before dropna, even if they are all None (won't be dropped by this)
        for col_ensure in ["ISIN", "Name of Instrument", "Market Value"]:
            if col_ensure not in df_clean.columns:
                df_clean[col_ensure] = None # Add if missing
        df_clean.dropna(subset=["ISIN", "Name of Instrument", "Market Value"], inplace=True)

        if df_clean.empty:
            print(f"⚠️ Sheet '{sheet_name}' became empty after dropping NA rows. Skipping.")
            return

        # Axis-specific Instrument Type Determination (as per legacy)
        # This overwrites any 'Type' column that might have come from mapping, or creates it.
        # The legacy logic: 'Debt' if yield != 0 else 'Equity or Equity related'
        df_clean['Type'] = df_clean['Yield'].apply(lambda x: 'Debt' if x != 0 else 'Equity or Equity related')

        # Numeric conversions and rounding for other relevant columns
        # Legacy Axis script implies these are the numeric columns (Quantity, Market Value, % to Net Assets)
        # Yield and Coupon are already numeric. 'Yield to call' is not in legacy Axis.
        numeric_cols = ['Quantity', 'Market Value', '% to Net Assets']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        df_clean = df_clean.round(2)

        df_clean["Scheme Name"] = fund_name
        df_clean["AMC"] = self.amc_name
        df_clean["Scheme ISIN"] = fund_isin if fund_isin else None

        for col in self.final_columns:
            if col not in df_clean.columns:
                df_clean[col] = None

        df_clean = df_clean[[col for col in self.final_columns if col in df_clean.columns]]

        self.full_data = pd.concat([self.full_data, df_clean], ignore_index=True) if not self.full_data.empty else df_clean
        print(f"✅ Successfully processed sheet: {sheet_name} for fund: {fund_name}")


class BandhanParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Bandhan Mutual Fund
        pass


class BankOfIndiaParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Bank of India Mutual Fund
        pass


class BarodaBNPParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Baroda BNP Paribas Mutual Fund
        pass


class CanaraRobecoParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Canara Robeco Mutual Fund
        pass


class DSPParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for DSP Mutual Fund
        pass


class EdelweissParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Edelweiss Mutual Fund
        pass


class FranklinTempletonParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Franklin Templeton India
        pass


class GrowwParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Groww Mutual Fund
        pass


class HDFCParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def _clean_fund_name(self, fund_name):
        """
        Cleans the fund name by removing unwanted characters and normalizing it.
        This method can be overridden in subclasses for specific fund name cleaning logic.
        """
        # Default implementation: strip whitespace and convert to lowercase
        cleaned_name = re.sub(r'\s+fund.*', ' fund', fund_name, flags=re.IGNORECASE)
        return cleaned_name.strip()
    

    def _get_fund_name(self, sheet_df):
        """
        Extracts the fund name from the sheet DataFrame.
        This method can be overridden in subclasses for specific fund name extraction logic.
        """
        # Default implementation: use the first non-empty cell in the first row
        name=sheet_df.head(0).columns[0]
        name= self._clean_fund_name(name)
        if name:
            return name
        else:
            print("⚠️ No fund name found in the sheet.")


    def process_sheet(self, datafile, sheet_name, sheet_df):
        print(f"\n🔍 Processing  → Sheet: {sheet_name}")
        
        fund_name = self._get_fund_name(sheet_df)
        print("Fund Name:", fund_name)
        AMC_NAME = self.amc_name
        
        if fund_name is not None and sheet_name:
                
                fund_isin = self._get_fund_isin(fund_name)
                print(f"\n🔍 Processing  → Sheet: {fund_name}, {fund_isin}")
                
                fund_isin = self._get_fund_isin(fund_name)

                header_row_idx = next(
                    (index for index, row in sheet_df.iterrows() if any("ISIN" in str(val) for val in row.dropna())),
                    None
                )
                if header_row_idx is None:
                    print(f"⚠️ Skipping {sheet_name} (No ISIN header found)")
                    return

                df_clean = pd.read_excel(datafile, sheet_name=sheet_name, skiprows=header_row_idx, dtype=str)
                df_clean.columns = df_clean.iloc[0]
                df_clean = df_clean[1:].reset_index(drop=True)
                df_clean = df_clean.loc[:, df_clean.columns.notna()]


                col_mapping=self.column_mapping  #obtain col maping from config to standardize column names


                  # Standardize column names
             
                df_clean=df_clean.rename(columns=col_mapping)

                #print(df_clean.columns) #to debug the column names


                df_clean.dropna(subset=["ISIN", "Name of Instrument", "Market Value"], inplace=True)
                
                df_clean['Type'] = df_clean['Industry'].apply(lambda x: 'Debt or related' if 'DEBT' in str(x).upper() else 'Equity or Equity related')
                
                df_clean = df_clean.round(2)
                df_clean["Scheme Name"] = fund_name
                df_clean["AMC"] = AMC_NAME
                df_clean["Scheme ISIN"] = fund_isin if fund_isin is not None else None


                # Standardize column names for outputs
                df_clean = df_clean[[col for col in self.final_columns if col in df_clean.columns]]

                self.full_data = pd.concat([self.full_data, df_clean], ignore_index=True) if not self.full_data.empty else df_clean

        


class HeliosParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Helios Mutual Fund
        pass


class HSBCParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for HSBC Mutual Fund
        pass


class InvescoParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Invesco Mutual Fund
        pass


class ITIParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for ITI Mutual Fund
        pass


class JMFinancialParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for JM Financial Mutual Fund
        pass


class KotakParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)
        self.kotak_scheme_map = {}
        self.kotak_scheme_map_file = None

    def _get_fund_name(self, sheet_df_ignored): # sheet_df is ignored for Kotak
        """
        Retrieves the full fund name for Kotak using the current sheet name (abbreviation)
        by looking it up in a map created from the "Scheme" sheet of the current datafile.
        Relies on self.current_datafile and self.current_sheet_name set by AMCPortfolioParser.
        """
        datafile = self.current_datafile
        sheet_name_key = self.current_sheet_name

        if sheet_name_key == "Scheme": # Do not process the "Scheme" sheet itself as a fund
            return None

        # Initialize or update scheme map if datafile changed or map not present
        if not self.kotak_scheme_map or self.kotak_scheme_map_file != datafile:
            try:
                # print(f"DEBUG: Reading Scheme sheet from {datafile} for Kotak scheme map.") # Optional debug
                fund_names_df = pd.read_excel(datafile, sheet_name="Scheme", dtype=str)

                if fund_names_df.empty:
                    print(f"⚠️ 'Scheme' sheet in {datafile} is empty.")
                    self.kotak_scheme_map = {}
                elif fund_names_df.iloc[0].isnull().all(): # Check if first row (potential header) is all NaN
                    print(f"⚠️ 'Scheme' sheet in {datafile} header row is missing or empty.")
                    # Attempt to use a default header if columns look like Abbreviations, Scheme Name
                    if len(fund_names_df.columns) >= 2 and 'Abbreviations' in fund_names_df.columns and 'Scheme Name' in fund_names_df.columns:
                         #This case means headers might be okay despite iloc[0] being NaN, proceed cautiously
                         pass # Let it try to use existing columns.
                    elif len(fund_names_df.columns) >=2 : # Try to assign default column names if possible
                        print(f"ℹ️ Trying to assign default columns 'Abbreviations', 'Scheme Name' to 'Scheme' sheet in {datafile}")
                        # This is risky, depends on actual structure.
                        # For now, let's assume first two columns are the ones we need if headers are bad.
                        # A more robust way would be to check for specific keywords in first few rows.
                        # Sticking to the original plan of using iloc[0] as header, then [1:] as data.
                        # If iloc[0] is bad, this will likely fail at 'Abbreviations' or 'Scheme Name' check.
                        # Fallback: if columns are just RangeIndex, try to use first two.
                        if isinstance(fund_names_df.columns, pd.RangeIndex) and len(fund_names_df.columns) >=2:
                            fund_names_df = fund_names_df.iloc[1:] # Skip what was likely a bad header
                            fund_names_df.columns = ['Abbreviations', 'Scheme Name'] + fund_names_df.columns[2:].tolist()


                # Find the header row that contains 'Abbreviations' or 'Scheme Name'
                header_row_idx = -1
                for i, row in fund_names_df.head(5).iterrows(): # Check first 5 rows
                    if any(col_val in ['Abbreviations', 'Scheme Name'] for col_val in row.dropna().astype(str).tolist()):
                        header_row_idx = i
                        break

                if header_row_idx != -1:
                    # Header found
                    fund_names_df.columns = fund_names_df.iloc[header_row_idx].str.strip()
                    fund_names_df = fund_names_df[header_row_idx + 1:].reset_index(drop=True)
                else:
                    # No specific header found, assume first row is header as per legacy
                    print(f"⚠️ Could not find specific header in 'Scheme' sheet of {datafile}. Assuming first row as header.")
                    if not fund_names_df.empty:
                        fund_names_df.columns = fund_names_df.iloc[0].str.strip()
                        fund_names_df = fund_names_df[1:].reset_index(drop=True)
                    else:
                        print(f"⚠️ 'Scheme' sheet in {datafile} is empty or header could not be processed.")
                        self.kotak_scheme_map = {}
                        self.kotak_scheme_map_file = datafile
                        return self.kotak_scheme_map.get(sheet_name_key)

                # Ensure required columns exist before creating the map
                if 'Abbreviations' in fund_names_df.columns and 'Scheme Name' in fund_names_df.columns:
                    # Drop rows where either 'Abbreviations' or 'Scheme Name' is NaN, as they can't be part of the map
                    fund_names_df.dropna(subset=['Abbreviations', 'Scheme Name'], inplace=True)
                    self.kotak_scheme_map = dict(zip(fund_names_df['Abbreviations'].str.strip(), fund_names_df['Scheme Name'].str.strip()))
                else:
                    print(f"⚠️ 'Abbreviations' or 'Scheme Name' column not found after header processing in 'Scheme' sheet of {datafile}.")
                    self.kotak_scheme_map = {}

                self.kotak_scheme_map_file = datafile
            except Exception as e:
                print(f"❌ Error reading or parsing 'Scheme' sheet from {datafile} for Kotak: {e}")
                self.kotak_scheme_map = {} # Ensure it's a dict
                self.kotak_scheme_map_file = None # Reset to allow reload attempt

        return self.kotak_scheme_map.get(sheet_name_key)

    def _clean_fund_name(self, fund_name):
        if not fund_name:
            return None
        cleaned_name = re.sub(r'\s+fund.*', ' fund', fund_name, flags=re.IGNORECASE)
        return cleaned_name.strip()

    def process_sheet(self, datafile, sheet_name, sheet_df):
        print(f"\n🔍 Processing Sheet for Kotak: {sheet_name} in file: {datafile}")

        # For Kotak, fund name is derived from sheet_name using the "Scheme" mapping
        fund_name_raw = self._get_fund_name(sheet_df) # sheet_df is ignored by Kotak's _get_fund_name

        if not fund_name_raw:
            print(f"ℹ️ Skipping sheet '{sheet_name}' for Kotak as its name was not found in the 'Scheme' sheet map or it is the 'Scheme' sheet itself.")
            return

        fund_name = self._clean_fund_name(fund_name_raw)
        if not fund_name: # Should not happen if fund_name_raw was valid, but as a safeguard
            print(f"⚠️ Skipping sheet '{sheet_name}' for Kotak due to invalid cleaned fund name.")
            return

        print(f"Found Fund Name for sheet '{sheet_name}': {fund_name}")
        fund_isin = self._get_fund_isin(fund_name)

        header_row_idx = next(
            (index for index, row in sheet_df.iterrows() if any("ISIN" in str(val) for val in row.dropna())),
            None
        )

        if header_row_idx is None:
            print(f"⚠️ Skipping sheet '{sheet_name}' for Kotak (No ISIN header found).")
            return

        df_clean = pd.read_excel(datafile, sheet_name=sheet_name, skiprows=header_row_idx, dtype=str)
        if df_clean.empty or df_clean.iloc[0].isnull().all():
            print(f"⚠️ Dataframe is empty or header is all NaN after skiprows for sheet '{sheet_name}'. Skipping.")
            return

        df_clean.columns = df_clean.iloc[0]
        df_clean = df_clean[1:].reset_index(drop=True)
        df_clean = df_clean.loc[:, df_clean.columns.notna()] # Remove completely NaN columns

        col_mapping = self.column_mapping
        df_clean = df_clean.rename(columns=col_mapping)

        # Data Cleaning specific to Kotak, aligned with legacy script logic

        # Ensure 'Type', 'Coupon', 'Yield' columns exist after mapping, adding them if they don't.
        # This is important because subsequent logic relies on them.
        # If they weren't in the source or mapped, they'll be filled with NaN or 0.
        if 'Type' not in df_clean.columns:
            df_clean['Type'] = pd.NA # Add as Pandas NA initially
        if 'Coupon' not in df_clean.columns:
            df_clean['Coupon'] = 0 # Legacy default
        if 'Yield' not in df_clean.columns:
            df_clean['Yield'] = 0 # Legacy default

        # Special handling for "Listed/Awaiting listing on Stock Exchange" in 'Coupon' column
        # This logic is ported from the legacy script, with correction for type propagation.
        if 'Coupon' in df_clean.columns and 'Type' in df_clean.columns:
            listing_indices = df_clean.index[df_clean['Coupon'].astype(str).str.contains("Listed/Awaiting listing on Stock Exchange", na=False)].tolist()
            for idx in listing_indices:
                if idx > 0: # Ensure we can access previous row
                    # Check if previous row's 'Type' is valid to propagate
                    # Legacy script used previous row's 'Coupon' which was likely an error. Using previous 'Type'.
                    prev_type = df_clean.at[df_clean.index[df_clean.index.get_loc(idx) - 1], 'Type']
                    if pd.notna(prev_type):
                         df_clean.at[idx, 'Type'] = prev_type
                    else:
                        # If previous type is also NA, this specific case might remain NA until ffill
                        pass # Or some other default if needed here
                # If idx is 0, this row's 'Type' won't be changed by this specific logic here, will be handled by ffill or default.

        # Fill 'Type' using forward fill (as per legacy)
        # This handles the "Listed/Awaiting..." cases correctly and propagates other types
        if 'Type' in df_clean.columns:
            df_clean['Type'] = df_clean['Type'].fillna(method='ffill')

        # Fill NA for Yield and Coupon (as per legacy)
        # Convert to numeric first, then fillna. This matches new parser's robust approach.
        df_clean['Yield'] = pd.to_numeric(df_clean['Yield'], errors='coerce').fillna(0)
        df_clean['Coupon'] = pd.to_numeric(df_clean['Coupon'], errors='coerce').fillna(0)

        # Drop rows that are crucial for identification or value if they are NA (as per legacy)
        df_clean.dropna(subset=["ISIN", "Name of Instrument", "Market Value"], inplace=True)
        if df_clean.empty:
            print(f"⚠️ Sheet '{sheet_name}' became empty after dropping NA rows. Skipping.")
            return

        # Instrument Type Determination Fallback (if 'Type' is still not resolved)
        # This is applied if 'Type' column is still all NaNs or has NaNs after ffill.
        if 'Type' not in df_clean.columns or df_clean['Type'].isnull().all():
            print(f"ℹ️ 'Type' column for Kotak in sheet '{sheet_name}' is missing or all NaN. Applying default logic.")
            df_clean = self._default_instrument_type_logic(df_clean)
        elif df_clean['Type'].isnull().any(): # If some NaNs remain after ffill (e.g. if first row was NA)
            print(f"ℹ️ 'Type' column for Kotak in sheet '{sheet_name}' has remaining NaNs. Applying default logic to fill them.")
            # Create a boolean series for rows where 'Type' is NaN
            nan_type_rows = df_clean['Type'].isnull()
            # Apply default logic only to these rows
            if nan_type_rows.any():
                 df_clean.loc[nan_type_rows, 'Type'] = self._default_instrument_type_logic(df_clean[nan_type_rows])['Type']


        # Convert relevant columns to numeric and round (as per legacy)
        # Legacy script implies these are the numeric columns it processes beyond Yield/Coupon.
        # "Yield to call" is in default final_columns, but not explicitly in Kotak legacy.
        numeric_cols = ['Quantity', 'Market Value', '% to Net Assets', 'Yield', 'Coupon']
        # 'Yield to call' will be handled by the generic numeric conversion if present from mapping.
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        df_clean = df_clean.round(2) # Round all numeric columns

        df_clean["Scheme Name"] = fund_name
        df_clean["AMC"] = self.amc_name
        df_clean["Scheme ISIN"] = fund_isin if fund_isin else None

        # Ensure all final columns are present
        for col in self.final_columns:
            if col not in df_clean.columns:
                df_clean[col] = None # Add missing final columns with None/NaN

        df_clean = df_clean[[col for col in self.final_columns if col in df_clean.columns]]

        self.full_data = pd.concat([self.full_data, df_clean], ignore_index=True) if not self.full_data.empty else df_clean
        print(f"✅ Successfully processed sheet: {sheet_name} for fund: {fund_name}")


class LICParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for LIC Mutual Fund
        pass


class MahindraManulifeParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Mahindra Manulife Mutual Fund
        pass


class MiraeAssetParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)
        # Mirae does not use a separate scheme map file based on legacy analysis; fund name extracted from sheet.

    def _get_fund_name(self, sheet_df):
        # Using the default fund name extraction logic from AMCPortfolioParser,
        # as it matches the legacy Mirae parser's approach.
        return self._default_fund_name_extraction(sheet_df)

    def _clean_fund_name(self, fund_name):
        if not fund_name:
            return None
        # Legacy Mirae specific cleaning: re.sub(r'\s+fund.*', ' fund', fund_name, flags=re.IGNORECASE)
        # This keeps " fund" at the end if present, rather than removing it.
        cleaned_name = re.sub(r'\s+fund.*', ' fund', fund_name, flags=re.IGNORECASE)
        return cleaned_name.strip()

    def process_sheet(self, datafile, sheet_name, sheet_df):
        print(f"\n🔍 Processing Sheet for Mirae Asset: {sheet_name} in file: {datafile}")

        fund_name_raw = self._get_fund_name(sheet_df)
        fund_name = self._clean_fund_name(fund_name_raw)

        if not fund_name:
            print(f"ℹ️ Skipping sheet '{sheet_name}' for Mirae Asset (Fund name not extracted).")
            return

        print(f"Found Fund Name for sheet '{sheet_name}': {fund_name}")
        fund_isin = self._get_fund_isin(fund_name)

        header_row_idx = next(
            (index for index, row in sheet_df.iterrows() if any("ISIN" in str(val) for val in row.dropna())),
            None
        )

        if header_row_idx is None:
            print(f"⚠️ Skipping sheet '{sheet_name}' for Mirae Asset (No ISIN header found).")
            return

        df_clean = pd.read_excel(datafile, sheet_name=sheet_name, skiprows=header_row_idx, dtype=str)
        if df_clean.empty or df_clean.iloc[0].isnull().all():
            print(f"⚠️ Dataframe is empty or header is all NaN after skiprows for sheet '{sheet_name}'. Skipping.")
            return

        df_clean.columns = df_clean.iloc[0] # Actual headers from file
        df_clean = df_clean[1:].reset_index(drop=True)
        df_clean = df_clean.loc[:, df_clean.columns.notna()] # Remove unnamed columns

        # ESG/Column count check (from legacy)
        # First, check for ESG in any original column name (before mapping)
        # These checks are on the columns as read from the file, after removing fully NaN columns.
        if any("ESG" in str(col_name).upper() for col_name in df_clean.columns): # Check against uppercase ESG
            print(f"⚠️ Skipping sheet '{sheet_name}' for Mirae Asset (ESG found in column names).")
            return

        # Legacy column count check: expected 7 data columns.
        # If more than 7 columns exist after dropping fully unnamed ones, it's an unexpected format.
        # Legacy code used >8 then assigned 7, which is a bit indirect. Direct check for >7 is cleaner.
        if len(df_clean.columns) > 7:
            print(f"⚠️ Skipping sheet '{sheet_name}' for Mirae Asset (Expected 7 data columns, found {len(df_clean.columns)}).")
            return

        # Apply column mapping from config
        col_mapping = self.column_mapping
        df_clean = df_clean.rename(columns=col_mapping)

        # Legacy Mirae parser expected 7 columns after all processing and assigned specific names.
        # The column_mapping should aim to produce these:
        # ["Name of Instrument", "ISIN", "Industry", "Quantity", "Market Value", "% to Net Assets", "Yield"]
        # Check column count after mapping, corresponding to legacy `len(df_clean.columns) > 8` (was 7 target + 1 dummy)
        # The legacy check `len(df_clean.columns) > 8` was before assigning fixed names.
        # A better check now is if all *expected mapped columns* are present.
        # For now, let's trust the mapping and proceed. If mapping is correct, we get the 7 expected columns.
        # The legacy code's check for len > 8 seems to be a filter for unexpected formats.
        # A more direct check would be against the number of expected columns from mapping.
        # Max columns expected by legacy was 7 for data.

        # Ensure essential columns exist after mapping
        expected_cols_from_legacy = ["Name of Instrument", "ISIN", "Industry", "Quantity", "Market Value", "% to Net Assets", "Yield"]
        # If any of these are missing after mapping, it's an issue.
        # Let's be flexible: if a column is missing, it will be filled with None later.

        # Data Cleaning
        df_clean.dropna(subset=["ISIN", "Name of Instrument", "Market Value"], inplace=True)
        if df_clean.empty:
            print(f"⚠️ Sheet '{sheet_name}' became empty after dropping NA rows. Skipping.")
            return

        # Ensure 'Yield' column exists before filling NA, add if missing (though it's in expected_cols_from_legacy)
        if 'Yield' not in df_clean.columns:
            df_clean['Yield'] = 0 # Add it if column_mapping didn't produce it
        df_clean['Yield'] = pd.to_numeric(df_clean['Yield'], errors='coerce').fillna(0)

        # Instrument Type Determination (using default yield-based logic)
        df_clean = self._default_instrument_type_logic(df_clean)

        # Numeric conversions and rounding
        # Mirae legacy only listed 7 columns, not including Coupon or Yield to call.
        numeric_cols = ['Quantity', 'Market Value', '% to Net Assets', 'Yield']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        df_clean = df_clean.round(2)

        df_clean["Scheme Name"] = fund_name
        df_clean["AMC"] = self.amc_name
        df_clean["Scheme ISIN"] = fund_isin if fund_isin else None

        # Ensure all final columns are present, adding missing ones as None
        # "Coupon" and "Yield to call" will be added here if not produced by mapping.
        for col in self.final_columns:
            if col not in df_clean.columns:
                df_clean[col] = None

        df_clean = df_clean[[col for col in self.final_columns if col in df_clean.columns]]

        self.full_data = pd.concat([self.full_data, df_clean], ignore_index=True) if not self.full_data.empty else df_clean
        print(f"✅ Successfully processed sheet: {sheet_name} for fund: {fund_name}")


class MotilalOswalParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Motilal Oswal Mutual Fund
        pass


class NaviParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Navi Mutual Fund
        pass


class NipponIndiaParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)
        self.nippon_scheme_map = {}
        self.nippon_scheme_map_file = None

    def _get_fund_name(self, sheet_df_ignored): # sheet_df is ignored
        datafile = self.current_datafile
        sheet_name_key = self.current_sheet_name

        if sheet_name_key == "INDEX": # Do not process the "INDEX" sheet itself
            return None

        if not self.nippon_scheme_map or self.nippon_scheme_map_file != datafile:
            try:
                fund_names_df = pd.read_excel(datafile, sheet_name="INDEX", dtype=str)
                if fund_names_df.empty:
                    print(f"⚠️ 'INDEX' sheet in {datafile} is empty.")
                    self.nippon_scheme_map = {}
                    self.nippon_scheme_map_file = datafile # Mark as processed to avoid re-reading empty
                    return self.nippon_scheme_map.get(sheet_name_key)

                # Legacy parser directly assigned columns: ["Short Name", "Scheme Name"]
                # This assumes the "INDEX" sheet has exactly two columns in that order, or
                # that the first two columns are what we need, regardless of actual headers.
                if fund_names_df.shape[1] >= 2:
                    # Use first two columns, assign standard names for map creation
                    fund_names_df = fund_names_df.iloc[:, [0, 1]]
                    fund_names_df.columns = ["Short Name", "Scheme Name"]
                    self.nippon_scheme_map = dict(zip(fund_names_df['Short Name'].str.strip(), fund_names_df['Scheme Name'].str.strip()))
                else:
                    print(f"⚠️ 'INDEX' sheet in {datafile} does not have at least two columns.")
                    self.nippon_scheme_map = {}

                self.nippon_scheme_map_file = datafile
            except Exception as e:
                print(f"❌ Error reading or parsing 'INDEX' sheet from {datafile} for Nippon India: {e}")
                self.nippon_scheme_map = {}
                self.nippon_scheme_map_file = None

        return self.nippon_scheme_map.get(sheet_name_key)

    def _clean_fund_name(self, fund_name):
        if not fund_name:
            return None
        # Using the same cleaning logic as legacy Mirae/Nippon
        cleaned_name = re.sub(r'\s+fund.*', ' fund', fund_name, flags=re.IGNORECASE)
        return cleaned_name.strip()

    def process_sheet(self, datafile, sheet_name, sheet_df):
        print(f"\n🔍 Processing Sheet for Nippon India: {sheet_name} in file: {datafile}")

        fund_name_raw = self._get_fund_name(sheet_df) # sheet_df is ignored

        if not fund_name_raw:
            print(f"ℹ️ Skipping sheet '{sheet_name}' for Nippon India (Fund name not found in INDEX or sheet is INDEX).")
            return

        fund_name = self._clean_fund_name(fund_name_raw)
        if not fund_name:
            print(f"⚠️ Skipping sheet '{sheet_name}' for Nippon India due to invalid cleaned fund name.")
            return

        print(f"Found Fund Name for sheet '{sheet_name}': {fund_name}")
        fund_isin = self._get_fund_isin(fund_name)

        header_row_idx = next(
            (index for index, row in sheet_df.iterrows() if any("ISIN" in str(val) for val in row.dropna())),
            None
        )

        if header_row_idx is None:
            print(f"⚠️ Skipping sheet '{sheet_name}' for Nippon India (No ISIN header found).")
            return

        df_clean = pd.read_excel(datafile, sheet_name=sheet_name, skiprows=header_row_idx, dtype=str)
        if df_clean.empty or df_clean.iloc[0].isnull().all():
            print(f"⚠️ Dataframe is empty or header is all NaN after skiprows for sheet '{sheet_name}'. Skipping.")
            return

        df_clean.columns = df_clean.iloc[0] # Actual headers from file
        df_clean = df_clean[1:].reset_index(drop=True)
        df_clean = df_clean.loc[:, df_clean.columns.notna()]

        # Apply column mapping from config.
        # Legacy parser assigned fixed column names: ["ISIN", "Name of Instrument", "Industry", "Quantity", "Market Value", "% to Net Assets", "Yield"]
        # The column_mapping in YAML should map actual file headers to these 7 target names.
        col_mapping = self.column_mapping
        df_clean = df_clean.rename(columns=col_mapping)

        # Ensure essential columns exist after mapping
        # Based on legacy, these are the 7 columns expected.
        expected_cols_from_legacy = ["ISIN", "Name of Instrument", "Industry", "Quantity", "Market Value", "% to Net Assets", "Yield"]
        # For flexibility, we'll add missing ones as None later, rather than erroring out if mapping isn't perfect.

        # Data Cleaning
        df_clean.dropna(subset=["ISIN", "Name of Instrument", "Market Value"], inplace=True)
        if df_clean.empty:
            print(f"⚠️ Sheet '{sheet_name}' became empty after dropping NA rows. Skipping.")
            return

        if 'Yield' not in df_clean.columns:
            df_clean['Yield'] = 0 # Add if missing
        df_clean['Yield'] = pd.to_numeric(df_clean['Yield'], errors='coerce').fillna(0)

        # Instrument Type Determination
        # Legacy used 'Debt' if yield != 0. Default logic uses 'Debt or related'.
        # Using default logic for consistency unless 'Debt' specifically is required.
        df_clean = self._default_instrument_type_logic(df_clean)

        # Numeric conversions and rounding
        numeric_cols = ['Quantity', 'Market Value', '% to Net Assets', 'Yield']
        # Coupon and Yield to call are not in legacy Nippon data.
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        df_clean = df_clean.round(2)

        df_clean["Scheme Name"] = fund_name
        df_clean["AMC"] = self.amc_name
        df_clean["Scheme ISIN"] = fund_isin if fund_isin else None

        for col in self.final_columns:
            if col not in df_clean.columns:
                df_clean[col] = None

        df_clean = df_clean[[col for col in self.final_columns if col in df_clean.columns]]

        self.full_data = pd.concat([self.full_data, df_clean], ignore_index=True) if not self.full_data.empty else df_clean
        print(f"✅ Successfully processed sheet: {sheet_name} for fund: {fund_name}")


class NJParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for NJ Mutual Fund
        pass


class PGIMIndiaParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for PGIM India Mutual Fund
        pass


class PPFASParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)
        self.ppfas_scheme_map = {}
        self.ppfas_scheme_map_file = None

    def _get_fund_name(self, sheet_df_ignored): # sheet_df is ignored
        datafile = self.current_datafile
        sheet_name_key = self.current_sheet_name

        if sheet_name_key == "Index": # Do not process the "Index" sheet itself
            return None

        if not self.ppfas_scheme_map or self.ppfas_scheme_map_file != datafile:
            try:
                fund_names_df = pd.read_excel(datafile, sheet_name="Index", dtype=str)
                if fund_names_df.empty:
                    print(f"⚠️ 'Index' sheet in {datafile} is empty.")
                    self.ppfas_scheme_map = {}
                    self.ppfas_scheme_map_file = datafile # Mark as processed
                    return self.ppfas_scheme_map.get(sheet_name_key)

                # Legacy parser assumes 'Short Name' and 'Scheme Name' columns in "Index" sheet
                if 'Short Name' in fund_names_df.columns and 'Scheme Name' in fund_names_df.columns:
                    self.ppfas_scheme_map = dict(zip(fund_names_df['Short Name'].str.strip(), fund_names_df['Scheme Name'].str.strip()))
                # Fallback if direct names not found - try first two columns if sheet structure is consistent
                elif fund_names_df.shape[1] >= 2:
                    print(f"ℹ️ 'Short Name'/'Scheme Name' not found in 'Index' sheet of {datafile} by header. Using first two columns.")
                    fund_names_df.columns = ["Short Name", "Scheme Name"] + fund_names_df.columns[2:].tolist() # Rename first two
                    self.ppfas_scheme_map = dict(zip(fund_names_df['Short Name'].str.strip(), fund_names_df['Scheme Name'].str.strip()))
                else:
                    print(f"⚠️ 'Index' sheet in {datafile} does not have 'Short Name'/'Scheme Name' columns or at least two columns.")
                    self.ppfas_scheme_map = {}

                self.ppfas_scheme_map_file = datafile
            except Exception as e:
                print(f"❌ Error reading or parsing 'Index' sheet from {datafile} for PPFAS: {e}")
                self.ppfas_scheme_map = {}
                self.ppfas_scheme_map_file = None

        return self.ppfas_scheme_map.get(sheet_name_key)

    def _clean_fund_name(self, fund_name):
        if not fund_name:
            return None
        # Standard cleaning: remove trailing " fund", " scheme" etc. and extra spaces
        # PPFAS legacy script didn't show specific cleaning applied to fund name after map lookup
        cleaned_name = re.sub(r'\s+(fund|scheme).*', '', fund_name, flags=re.IGNORECASE)
        return cleaned_name.strip()

    def process_sheet(self, datafile, sheet_name, sheet_df):
        print(f"\n🔍 Processing Sheet for PPFAS: {sheet_name} in file: {datafile}")

        fund_name_raw = self._get_fund_name(sheet_df) # sheet_df is ignored

        if not fund_name_raw:
            print(f"ℹ️ Skipping sheet '{sheet_name}' for PPFAS (Fund name not found in Index or sheet is Index).")
            return

        fund_name = self._clean_fund_name(fund_name_raw)
        if not fund_name: # Should not happen if fund_name_raw was valid, but as a safeguard
            print(f"⚠️ Skipping sheet '{sheet_name}' for PPFAS due to invalid cleaned fund name.")
            return

        print(f"Found Fund Name for sheet '{sheet_name}': {fund_name}")
        fund_isin = self._get_fund_isin(fund_name)

        header_row_idx = next(
            (index for index, row in sheet_df.iterrows() if any("ISIN" in str(val) for val in row.dropna())),
            None
        )

        if header_row_idx is None:
            print(f"⚠️ Skipping sheet '{sheet_name}' for PPFAS (No ISIN header found).")
            return

        df_clean = pd.read_excel(datafile, sheet_name=sheet_name, skiprows=header_row_idx, dtype=str)
        if df_clean.empty or df_clean.iloc[0].isnull().all():
            print(f"⚠️ Dataframe is empty or header is all NaN after skiprows for sheet '{sheet_name}'. Skipping.")
            return

        df_clean.columns = df_clean.iloc[0] # Actual headers from file
        df_clean = df_clean[1:].reset_index(drop=True)
        df_clean = df_clean.loc[:, df_clean.columns.notna()]

        # Apply column mapping from config.
        # Legacy parser assigned: ["Name of Instrument", "ISIN", "Industry", "Quantity", "Market Value (Rs.in Lacs)", "% to Net Assets", "Yield", "Yield 2"]
        # Then renamed "Market Value (Rs.in Lacs)" to "Market Value"
        # And dropped "Yield 2"
        # So, column_mapping should produce these names, with "Market Value (Rs.in Lacs)" being mapped to "Market Value" directly if possible,
        # or mapped to "Market Value (Rs.in Lacs)" then renamed (but direct mapping is cleaner).
        # And "Yield 2" should be mapped if it exists.
        col_mapping = self.column_mapping
        df_clean = df_clean.rename(columns=col_mapping)

        # Drop "Yield 2" if it exists after mapping (as per legacy)
        if "Yield 2" in df_clean.columns:
            df_clean = df_clean.drop(columns=["Yield 2"])

        # Data Cleaning
        df_clean.dropna(subset=["ISIN", "Name of Instrument", "Market Value"], inplace=True)
        if df_clean.empty:
            print(f"⚠️ Sheet '{sheet_name}' became empty after dropping NA rows. Skipping.")
            return

        if 'Yield' not in df_clean.columns:
            df_clean['Yield'] = 0 # Add if missing (legacy implies it's expected)
        df_clean['Yield'] = pd.to_numeric(df_clean['Yield'], errors='coerce').fillna(0)

        # Instrument Type Determination
        df_clean = self._default_instrument_type_logic(df_clean)

        # Numeric conversions and rounding
        numeric_cols = ['Quantity', 'Market Value', '% to Net Assets', 'Yield']
        # Coupon and Yield to call are not in legacy PPFAS data.
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        df_clean = df_clean.round(2)

        df_clean["Scheme Name"] = fund_name
        df_clean["AMC"] = self.amc_name # Should be "PPFAS Mutual Fund" or as per config
        df_clean["Scheme ISIN"] = fund_isin if fund_isin else None

        for col in self.final_columns:
            if col not in df_clean.columns:
                df_clean[col] = None

        df_clean = df_clean[[col for col in self.final_columns if col in df_clean.columns]]

        self.full_data = pd.concat([self.full_data, df_clean], ignore_index=True) if not self.full_data.empty else df_clean
        print(f"✅ Successfully processed sheet: {sheet_name} for fund: {fund_name}")


class QuantParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)
        self.max_columns_allowed = config.get("max_columns_allowed", 10) # From legacy analysis

    def _get_fund_name(self, sheet_df):
        # Using the default fund name extraction logic from AMCPortfolioParser,
        # as it matches the legacy Quant parser's approach.
        return self._default_fund_name_extraction(sheet_df)

    def _clean_fund_name(self, fund_name):
        if not fund_name:
            return None
        # Standard cleaning, as legacy Quant script didn't apply its clean_fund_name function.
        cleaned_name = re.sub(r'\s+(fund|scheme).*', '', fund_name, flags=re.IGNORECASE)
        return cleaned_name.strip()

    def process_sheet(self, datafile, sheet_name, sheet_df):
        print(f"\n🔍 Processing Sheet for Quant: {sheet_name} in file: {datafile}")

        fund_name_raw = self._get_fund_name(sheet_df)
        fund_name = self._clean_fund_name(fund_name_raw)

        if not fund_name:
            print(f"ℹ️ Skipping sheet '{sheet_name}' for Quant (Fund name not extracted).")
            return

        print(f"Found Fund Name for sheet '{sheet_name}': {fund_name}")
        fund_isin = self._get_fund_isin(fund_name)

        header_row_idx = next(
            (index for index, row in sheet_df.iterrows() if any("ISIN" in str(val) for val in row.dropna())),
            None
        )

        if header_row_idx is None:
            print(f"⚠️ Skipping sheet '{sheet_name}' for Quant (No ISIN header found).")
            return

        df_clean = pd.read_excel(datafile, sheet_name=sheet_name, skiprows=header_row_idx, dtype=str)
        if df_clean.empty or df_clean.iloc[0].isnull().all():
            print(f"⚠️ Dataframe is empty or header is all NaN after skiprows for sheet '{sheet_name}'. Skipping.")
            return

        df_clean.columns = df_clean.iloc[0] # Actual headers from file
        df_clean = df_clean[1:].reset_index(drop=True)
        df_clean = df_clean.loc[:, df_clean.columns.notna()]

        # Column count filter (from legacy)
        if len(df_clean.columns) > self.max_columns_allowed:
            print(f"⚠️ Skipping sheet '{sheet_name}' for Quant (Too many columns: {len(df_clean.columns)}, max allowed: {self.max_columns_allowed}).")
            return

        # Apply column mapping from config.
        # Legacy parser assigned: ["SR","ISIN","Name of Instrument", "Rating", "Industry", "Quantity", "Market Value", "% to Net Assets", "Yield"]
        # The column_mapping in YAML should map actual file headers to these 9 target names.
        col_mapping = self.column_mapping
        df_clean = df_clean.rename(columns=col_mapping)

        # Drop "SR" column if it exists after mapping (as per legacy)
        if "SR" in df_clean.columns:
            df_clean = df_clean.drop(columns=["SR"])

        # Data Cleaning
        # Ensure essential columns exist after mapping
        # Based on legacy (after dropping SR): "ISIN", "Name of Instrument", "Rating", "Industry", "Quantity", "Market Value", "% to Net Assets", "Yield"]

        df_clean.dropna(subset=["ISIN", "Name of Instrument", "Market Value"], inplace=True)
        if df_clean.empty:
            print(f"⚠️ Sheet '{sheet_name}' became empty after dropping NA rows. Skipping.")
            return

        if 'Yield' not in df_clean.columns:
            df_clean['Yield'] = 0 # Add if missing
        df_clean['Yield'] = pd.to_numeric(df_clean['Yield'], errors='coerce').fillna(0)

        # Handle 'Rating' column - if not in final_columns, it will be dropped.
        # If it should be merged with 'Industry', mapping should handle it: e.g. {"Rating": "Industry", "Industry": "Industry"}
        # For now, assume 'Rating' is a distinct column that might or might not be in final_columns.
        if 'Rating' not in df_clean.columns:
            df_clean['Rating'] = None # Add if missing, so it doesn't fail if final_columns expects it

        # Instrument Type Determination
        df_clean = self._default_instrument_type_logic(df_clean)

        # Numeric conversions and rounding
        numeric_cols = ['Quantity', 'Market Value', '% to Net Assets', 'Yield']
        # Coupon and Yield to call are not in legacy Quant data.
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        df_clean = df_clean.round(2)

        df_clean["Scheme Name"] = fund_name
        df_clean["AMC"] = self.amc_name
        df_clean["Scheme ISIN"] = fund_isin if fund_isin else None

        for col in self.final_columns:
            if col not in df_clean.columns:
                df_clean[col] = None

        df_clean = df_clean[[col for col in self.final_columns if col in df_clean.columns]]

        self.full_data = pd.concat([self.full_data, df_clean], ignore_index=True) if not self.full_data.empty else df_clean
        print(f"✅ Successfully processed sheet: {sheet_name} for fund: {fund_name}")


class QuantumParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Quantum Mutual Fund
        pass


class SBIParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)
        self.sbi_scheme_map = {}
        self.sbi_scheme_map_file = None
        # Specific columns to drop for SBI, loaded from config
        # Handle potential string newlines in YAML by replacing with actual newlines for matching
        self.sbi_columns_to_drop = [col_name.replace("\\n", "\n") for col_name in config.get("sbi_columns_to_drop", [])]


    def _get_fund_name(self, sheet_df_ignored): # sheet_df is ignored
        datafile = self.current_datafile
        sheet_name_key = self.current_sheet_name

        if sheet_name_key == "Index": # Do not process the "Index" sheet itself
            return None

        if not self.sbi_scheme_map or self.sbi_scheme_map_file != datafile:
            try:
                # Legacy parser: header is iloc[1], data from iloc[2:]
                # header=0 would read the first row as header, then we skip one more row for data.
                # However, legacy explicitly sets columns from iloc[1].
                # Safest to read without header then process.
                fund_names_df_full = pd.read_excel(datafile, sheet_name="Index", header=None, dtype=str)
                if fund_names_df_full.empty:
                    print(f"⚠️ 'Index' sheet in {datafile} is empty.")
                    self.sbi_scheme_map = {}
                    self.sbi_scheme_map_file = datafile # Mark as processed
                    return self.sbi_scheme_map.get(sheet_name_key)

                if fund_names_df_full.shape[0] < 2: # Need at least 2 rows for header and data
                    print(f"⚠️ 'Index' sheet in {datafile} has insufficient rows to determine header and data (requires at least 2 rows).")
                    self.sbi_scheme_map = {}
                else:
                    header = fund_names_df_full.iloc[1] # Second row is the header
                    fund_names_df_data = fund_names_df_full[2:].copy() # Data starts from the third row
                    fund_names_df_data.columns = header

                    if 'Scheme Short code' in fund_names_df_data.columns and 'Scheme Name' in fund_names_df_data.columns:
                        self.sbi_scheme_map = dict(zip(fund_names_df_data['Scheme Short code'].str.strip(), fund_names_df_data['Scheme Name'].str.strip()))
                    else:
                        print(f"⚠️ 'Scheme Short code' or 'Scheme Name' not found in 'Index' sheet columns of {datafile} (expected after using row 1 as header).")
                        self.sbi_scheme_map = {}

                self.sbi_scheme_map_file = datafile
            except Exception as e:
                print(f"❌ Error reading or parsing 'Index' sheet from {datafile} for SBI: {e}")
                self.sbi_scheme_map = {}
                self.sbi_scheme_map_file = None

        return self.sbi_scheme_map.get(sheet_name_key)

    def _clean_fund_name(self, fund_name):
        if not fund_name:
            return None
        cleaned_name = re.sub(r'\s+fund.*', ' fund', fund_name, flags=re.IGNORECASE)
        return cleaned_name.strip()

    def process_sheet(self, datafile, sheet_name, sheet_df):
        print(f"\n🔍 Processing Sheet for SBI: {sheet_name} in file: {datafile}")

        fund_name_raw = self._get_fund_name(sheet_df) # sheet_df is ignored

        if not fund_name_raw:
            print(f"ℹ️ Skipping sheet '{sheet_name}' for SBI (Fund name not found in Index or sheet is Index).")
            return

        fund_name = self._clean_fund_name(fund_name_raw)
        if not fund_name:
            print(f"⚠️ Skipping sheet '{sheet_name}' for SBI due to invalid cleaned fund name.")
            return

        print(f"Found Fund Name for sheet '{sheet_name}': {fund_name}")
        fund_isin = self._get_fund_isin(fund_name)

        header_row_idx = next(
            (index for index, row in sheet_df.iterrows() if any("ISIN" in str(val) for val in row.dropna())),
            None
        )

        if header_row_idx is None:
            print(f"⚠️ Skipping sheet '{sheet_name}' for SBI (No ISIN header found).")
            return

        df_clean = pd.read_excel(datafile, sheet_name=sheet_name, skiprows=header_row_idx, dtype=str)
        if df_clean.empty or df_clean.iloc[0].isnull().all():
            print(f"⚠️ Dataframe is empty or header is all NaN after skiprows for sheet '{sheet_name}'. Skipping.")
            return

        df_clean.columns = df_clean.iloc[0]
        df_clean = df_clean[1:].reset_index(drop=True)
        df_clean = df_clean.loc[:, df_clean.columns.notna()]

        # Drop specific SBI columns (loaded from config) before general mapping
        # Exact column names from legacy analysis, including newlines if that's how pandas reads them.
        cols_to_drop_actually_present = [col for col in self.sbi_columns_to_drop if col in df_clean.columns]
        if cols_to_drop_actually_present:
            df_clean = df_clean.drop(columns=cols_to_drop_actually_present)
            print(f"ℹ️ Dropped specific SBI columns: {cols_to_drop_actually_present} from sheet '{sheet_name}'")

        col_mapping = self.column_mapping
        df_clean = df_clean.rename(columns=col_mapping)

        df_clean.dropna(subset=["ISIN", "Name of Instrument", "Market Value"], inplace=True)
        if df_clean.empty:
            print(f"⚠️ Sheet '{sheet_name}' became empty after dropping NA rows. Skipping.")
            return

        if 'Yield' not in df_clean.columns: # Ensure Yield column exists post-mapping
            df_clean['Yield'] = 0
        df_clean['Yield'] = pd.to_numeric(df_clean['Yield'], errors='coerce').fillna(0)

        # SBI Legacy Type: 'Debt' if yield != 0 else 'Equity or Equity related'
        df_clean['Type'] = df_clean['Yield'].apply(lambda x: 'Debt' if x != 0 else 'Equity or Equity related')

        numeric_cols = ['Quantity', 'Market Value', '% to Net Assets', 'Yield']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        df_clean = df_clean.round(2)

        df_clean["Scheme Name"] = fund_name
        df_clean["AMC"] = self.amc_name
        df_clean["Scheme ISIN"] = fund_isin if fund_isin else None

        for col in self.final_columns:
            if col not in df_clean.columns:
                df_clean[col] = None

        df_clean = df_clean[[col for col in self.final_columns if col in df_clean.columns]]

        self.full_data = pd.concat([self.full_data, df_clean], ignore_index=True) if not self.full_data.empty else df_clean
        print(f"✅ Successfully processed sheet: {sheet_name} for fund: {fund_name}")


class ShriramParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Shriram Mutual Fund
        pass


class SundaramParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Sundaram Mutual Fund
        pass


class TataParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Tata Mutual Fund
        pass


class TrustParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Trust Mutual Fund
        pass


class UnionParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Union Mutual Fund
        pass


class UTIParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for UTI Mutual Fund
        pass


class WhiteOakParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for WhiteOak Mutual Fund
        pass


class ZerodhaParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Zerodha Fund House
        pass









