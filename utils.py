class Utils:
    @staticmethod
    def read_txt_file(file_path):
        data_dict = {}
        current_title = None        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line in ["Location", "Business Model", "Investments", "Timeframe", "Customers"]:
                        current_title = line
                        data_dict[current_title] = []
                    elif current_title:
                        keywords = [kw.strip() for kw in line.split(",") if kw.strip()]
                        data_dict[current_title].extend(keywords)
                        
            return data_dict
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return {} 
    
