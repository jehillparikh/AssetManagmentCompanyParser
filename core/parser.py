from core.amcparser import AMCPortfolioParser
import pandas as pd
import re

class ICICIMFParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):

        print(f"\n🔍 Processing  → Sheet: {sheet_name}")
        fund_name = self._default_fund_name_extraction(sheet_df)
        

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

                print(df_clean.columns)

                if "Coupon" not in df_clean.columns:
                        df_clean.insert(3, 'Coupon','0')
                        df_clean['Coupon'] = 0

                    
                col_names=["Name of Instrument","ISIN", "Coupon" ,"Industry", "Quantity", "Market Value", "% to Net Assets", "Yield", "Yield to call"]
                    
                if len(df_clean.columns) >10:
                        print("⚠️ Skipping {sheet_name} (Too many columns) probably EGS fund)")
                        return

                df_clean.columns =col_names
                df_clean.dropna(subset=["ISIN", "Name of Instrument", "Market Value"], inplace=True)


                #Just a simple logic to determine the type of instrument need to update later TODO

                df_clean[['Yield']] = df_clean[['Yield']].fillna(value=0)
                df_clean['Type'] = df_clean['Yield'].apply(lambda x: 'Debt or related' if x != 0 else 'Equity or Equity related')
                    
                df_clean = df_clean.round(2)
                df_clean["Scheme Name"] = fund_name
                df_clean["AMC"] = self.amc_name
                df_clean["FUND_ISIN"] = fund_isin if fund_isin is not None else None
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

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Axis Mutual Fund
        pass


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
        amc_norm = self.amc_name.strip().lower()
        top6 = sheet_df.head(1).astype(str)
        for t in top6:
            print(self._clean_fund_name(t))


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


                df_clean.columns = ["Name of Instrument", "ISIN", "Industry", "Quantity", "Market Value", "% to Net Assets"]
                df_clean.dropna(subset=["ISIN", "Name of Instrument", "Market Value"], inplace=True)
                
                df_clean['Type'] = df_clean['Industry'].apply(lambda x: 'Debt or related' if 'DEBT' in str(x).upper() else 'Equity or Equity related')
                
                df_clean = df_clean.round(2)
                df_clean["Scheme Name"] = fund_name
                df_clean["AMC"] = AMC_NAME
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

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Kotak Mutual Fund
        pass


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

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Mirae Asset Mutual Fund
        pass


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

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Nippon India Mutual Fund
        pass


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

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for PPFAS Mutual Fund
        pass


class QuantParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Quant Mutual Fund
        pass


class QuantumParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for Quantum Mutual Fund
        pass


class SBIParser(AMCPortfolioParser):
    def __init__(self, config):
        super().__init__(config=config)

    def process_sheet(self, datafile, sheet_name, sheet_df):
        # TODO: Implement the specific cleaning logic for SBI Mutual Fund
        pass


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









